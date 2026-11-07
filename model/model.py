import math
import torch.nn.init as init
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

















class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,

        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)







        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func


        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from torch.nn import functional as F

from transformers.activations import ACT2FN


class RMSNorm(nn.Module):


    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))


    def _norm(self, x: torch.Tensor):




        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


    def forward(self, x):


        return self.weight * self._norm(x.float()).type_as(x)










def precompute_freqs(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                     rope_scaling: Optional[dict] = None):








    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:








        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
            rope_scaling.get("attention_factor", 1.0),
        )



        if end / orig_max > 1.0:




            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))







            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )







            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1,
            )






            freqs = freqs * (1 - ramp + ramp / factor)



    t = torch.arange(end, device=freqs.device)




    freqs = torch.outer(t, freqs).float()



    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin



def apply_rotary_pos_emb(q,
                         k,
                         cos, sin,
                         position_ids=None,
                         unsqueeze_dim=1):




    def rotate_half(x):



        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)










    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)

    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:



    bs, slen, num_key_value_heads, head_dim = x.shape

    if n_rep == 1:
        return x




    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):

    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads




        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)

        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)



        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(F, "scaled_dot_product_attention") and args.flash_attention

    def forward(self, x: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):





        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)


        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)



        cos, sin = position_embeddings

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)


        if past_key_value is not None:

            xk = torch.cat([past_key_value[0], xk], dim=1)

            xv = torch.cat([past_key_value[1], xv], dim=1)


        past_kv = (xk, xv) if use_cache else None



        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )




        if (self.flash and seq_len > 1 and past_key_value is None
                and (attention_mask is None or torch.all(attention_mask == 1))):



            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:



            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)




            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=scores.device), diagonal=1,
            )

            if attention_mask is not None:

                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

                scores = scores + extended_attention_mask



            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

            scores = self.attn_dropout(scores)

            output = scores @ xv





        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        output = self.resid_dropout(self.o_proj(output))


        return output, past_kv




class MoEGate(nn.Module):
    def __init__(self, args: MokioMindConfig):

        super().__init__()
        self.config = args

        self.top_k = args.num_experts_per_tok

        self.scoring_func = args.scoring_func
        self.alpha = args.aux_loss_alpha

        self.seq_aux = args.seq_aux
        self.n_routed_experts = args.n_routed_experts
        self.norm_topk_prob = args.norm_topk_prob
        self.gating_dim = args.hidden_size

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

        self.reset_parameters()

    def reset_parameters(self):


        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):


        bsz, seq_len, h= hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)


        logits = F.linear(hidden_states, self.weight, None)


        if self.scoring_func == 'softmax':


            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(f"不支持的MoE打分函数: {self.scoring_func!r}")


        topk_weight, topk_idx = torch.topk(
            scores, self.top_k, dim =-1, sorted=False
        )


        if self.top_k > 1 and self.norm_topk_prob:


            denominator = (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )

            topk_weight = topk_weight / denominator





        if self.training and self.alpha >0:
            scores_for_aux = scores
            aux_topk = self.top_k

            topk_idx_for_aux_loss = topk_idx.view(bsz, seq_len, -1)


            if self.seq_aux:

                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                topk_idx_for_aux_loss = topk_idx_for_aux_loss.reshape(
                    bsz, seq_len * aux_topk
                )
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device = hidden_states.device
                )


                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones_like(topk_idx_for_aux_loss, dtype=ce.dtype)
                )


                ce = ce.div(
                    seq_len*aux_topk/self.n_routed_experts
                )



                aux_loss = (ce*scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean()*self.alpha

            else:


                mask_ce =F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )

                ce=mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)

                fi = ce*self.n_routed_experts

                aux_loss = (fi*Pi).sum()*self.alpha

        else:

            aux_loss = scores.new_zeros(())


        return topk_idx, topk_weight, aux_loss


class MoeFeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x


        orig_shape = x.shape



        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.reshape(-1, x.shape[-1])


        flat_topk_idx = topk_idx.view(-1)

        if self.training:


            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)


            y = torch.empty_like(x)

            for i, expert in enumerate(self.experts):

                mask = flat_topk_idx == i


                y[mask] = expert(x[mask]).to(dtype=y.dtype)


            y = (
                y.view(*topk_weight.shape, -1)
                * topk_weight.unsqueeze(-1)
            ).sum(dim=1)


            y = y.to(dtype=identity.dtype).reshape(orig_shape)
        else:

            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1))
            y = y.view(*orig_shape)



        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity).to(dtype=y.dtype)


        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weight):




        expert_cache = torch.zeros_like(x, dtype=flat_expert_weight.dtype)


        idxs = flat_expert_indices.argsort()


        tokens_per_expert = flat_expert_indices.bincount(minlength=len(self.experts)).cumsum(0).tolist()


        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):

            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[i]


            exp_token_idx = token_idxs[start_idx:end_idx]


            expert_tokens = x[exp_token_idx]

            expert_out = expert(expert_tokens).to(dtype=x.dtype)

            weighted = expert_out * flat_expert_weight[idxs[start_idx:end_idx]]

            expert_cache.index_add_(0, exp_token_idx, weighted)

        return expert_cache.to(dtype=x.dtype)










class FeedForward(nn.Module):


    def __init__(self, args: MokioMindConfig):
        super().__init__()


        if args.intermediate_size is None:


            intermediate_size = int(args.hidden_size*8/3)



            args.intermediate_size = 64*((intermediate_size+64-1)//64)




        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)


        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)


        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)



        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):










        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))





class MokioMindBlock(nn.Module):


    def __init__(self, layer_id:int , config: MokioMindConfig):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.self_attn = Attention(config)


        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = (FeedForward(config) if not config.use_moe else MoeFeedForward(config))

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask = None):








        residual = hidden_states



        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )


        hidden_states = hidden_states + residual




        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


        return hidden_states, present_key_value



class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()


        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)



        self.layer = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(config.num_hidden_layers)]
        )


        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


        freqs_cos, freqs_sin = precompute_freqs(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )



        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids:Optional[torch.Tensor] = None,
            attention_mask:Optional[torch.Tensor] = None,
            past_key_values:Optional[Tuple[Tuple[torch.Tensor]]] = None,
            use_cache:bool = False,
            **kwargs
    ) -> torch.Tensor:




        batch_size, seq_len = input_ids.shape



        if hasattr(past_key_values, 'layers'):
            past_key_values = None


        past_key_values = past_key_values or [None]*len(self.layer)


        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))



        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )



        presents = []
        aux_loss = hidden_states.new_zeros(())



        for layer_idx, (layer, past_key_value) in enumerate(
                zip(self.layer, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )


            presents.append(present)

            aux_loss = aux_loss + getattr(layer.mlp, "aux_loss", hidden_states.new_zeros(()))


        hidden_states = self.norm(hidden_states)


        return hidden_states, presents, aux_loss

class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):


    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        self.config = config
        super().__init__(config)

        self.model = MokioMindModel(config)



        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size,
            bias = False
        )




        self.model.embed_tokens.weight = self.lm_head.weight


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor]=0,
            **args):





        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )






        slice_indices = (
            slice(-logits_to_keep,None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )



        logits = self.lm_head(hidden_states if labels is not None else hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:


            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()





            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ) / shift_labels.ne(-100).sum().clamp_min(1)




        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits if labels is None else logits[:, slice_indices, :],
            past_key_values=past_key_values if use_cache else None,
        )

        output["last_hidden_state"] = hidden_states

        output["aux_loss"] = aux_loss
        return output

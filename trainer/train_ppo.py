import os
import sys


__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model import MokioMindConfig, MokioMindForCausalLM
from dataset.lm_dataset import RLHFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)

warnings.filterwarnings("ignore")




class CriticModel(MokioMindForCausalLM):

    def __init__(self, params):
        super().__init__(params)

        self.value_head = nn.Linear(params.hidden_size, 1)


    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        hidden_states = outputs[0]

        values = self.value_head(hidden_states).squeeze(-1)
        return values




def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):

    def reasoning_model_reward(rewards):

        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"

        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"


        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [
            re.match(pattern2, response, re.S) for response in responses
        ]


        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)

        rewards += torch.tensor(format_rewards, device=args.device)


        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards


    rewards = torch.zeros(len(responses), device=args.device)

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():

        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)

            messages = [
                {"role": role, "content": content.strip()} for role, content in matches
            ]


            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(
                reward_tokenizer, tmp_chat
            )


            scale = 3.0
            score = max(min(score, scale), -scale)

            if args.reasoning == 1:
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                if answer_match:
                    answer_content = answer_match.group(1).strip()

                    tmp_chat = messages + [
                        {"role": "assistant", "content": answer_content}
                    ]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)


                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)

        rewards += reward_model_scores

    return rewards




def ppo_train_epoch(
        epoch,
        loader,
        iters,
        old_actor_model,
        ref_model,
        actor_scheduler,
        critic_scheduler,
        reward_model,
        reward_tokenizer,
        start_step=0,
        wandb=None,
):

    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        ).to(args.device)



        prompt_width = enc.input_ids.shape[1]

        with torch.no_grad():
            model_for_gen = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )


            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )



        responses_text = [
            tokenizer.decode(gen_out[i, prompt_width :], skip_special_tokens=True)
            for i in range(len(prompts))
        ]



        rewards = calculate_rewards(
            prompts, responses_text, reward_model, reward_tokenizer
        )






        response_ids = gen_out[:, prompt_width:]
        is_eos = response_ids.eq(tokenizer.eos_token_id)
        response_mask = (is_eos.cumsum(dim=1) - is_eos.long()).eq(0).long()
        full_mask = torch.cat([enc.attention_mask, response_mask], dim=1)


        value_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)


        last_indices = prompt_width + response_mask.sum(dim=1) - 1


        values = value_seq[torch.arange(len(last_indices), device=gen_out.device), last_indices]



        advantages = rewards - values.detach()




        logits = actor_model(
            input_ids=gen_out, attention_mask=full_mask
        ).logits


        labels = gen_out[:, 1:].clone()


        logp_tokens = (
            F.log_softmax(logits[:, :-1, :], dim=-1)
            .gather(2, labels.unsqueeze(-1))
            .squeeze(-1)
        )
        seq_len = gen_out.size(1) - 1



        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(
            0
        ) >= (prompt_width - 1)


        final_mask = resp_mask & full_mask[:, 1:].bool()



        actor_logp = (logp_tokens * final_mask).sum(dim=1)




        with torch.no_grad():
            old_logits = old_actor_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits
            old_logp_tokens = (
                F.log_softmax(old_logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )

            old_logp = (old_logp_tokens * final_mask).sum(dim=1)


            ref_logits = ref_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits
            ref_logp_tokens = (
                F.log_softmax(ref_logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )

            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)



        kl = (actor_logp - old_logp).mean()


        kl_ref = (actor_logp - ref_logp).mean()


        ratio = torch.exp(actor_logp - old_logp)



        surr1 = ratio * advantages
        surr2 = (

                torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
                * advantages
        )

        policy_loss = -torch.min(surr1, surr2).mean()



        value_loss = F.mse_loss(values, rewards)


        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref

        (loss / args.accumulation_steps).backward()



        if step % args.accumulation_steps == 0 or step == iters:
            remainder = step % args.accumulation_steps
            if remainder:
                correction = args.accumulation_steps / remainder
                for mdl in (actor_model, critic_model):
                    for parameter in mdl.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

            clip_grad_norm_(actor_model.parameters(), args.grad_clip)

            clip_grad_norm_(critic_model.parameters(), args.grad_clip)

            actor_optimizer.step()

            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()

            actor_optimizer.zero_grad()

            critic_optimizer.zero_grad()


        if is_main_process() and (step % args.log_interval == 0 or step == iters):

            response_ids = gen_out[:, enc.input_ids.shape[1] :]
            is_eos = response_ids == tokenizer.eos_token_id
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)

            lengths = torch.where(
                has_eos,
                eos_indices + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device),
                )
            avg_len = lengths.float().mean()


            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]["lr"]
            critic_lr = critic_optimizer.param_groups[0]["lr"]

            if wandb is not None:
                wandb.log(
                    {
                        "actor_loss": actor_loss_val,
                        "critic_loss": critic_loss_val,
                        "reward": reward_val,
                        "kl": kl_val,
                        "kl_ref": kl_ref_val,
                        "avg_response_len": avg_len_val,
                        "actor_lr": actor_lr,
                    }
                )

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}"
            )


        if step % args.update_old_actor_freq == 0:
            state_dict = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )

            old_actor_model.load_state_dict(
                {k: v.detach().cpu() for k, v in state_dict.items()}
            )
            old_actor_model.to(args.device)


        save_boundary = step % args.accumulation_steps == 0 or step == iters
        crossed_save_interval = step // args.save_interval > (step - args.accumulation_steps) // args.save_interval
        if (step == iters or (save_boundary and crossed_save_interval)) and is_main_process():
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            actor_state = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )

            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)



            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=actor_model,
                optimizer=actor_optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=CHECKPOINT_DIR,
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )
            actor_model.train()


if __name__ == "__main__":
    """
    PPO主函数：近端策略优化脚本的入口点

    📚 PPO训练架构：
    1. Actor模型：生成策略，输出动作概率
    2. Critic模型：价值函数，估计状态价值
    3. Reward模型：奖励函数，评估生成质量
    4. Old Actor：用于重要性采样的旧策略
    5. Reference：用于KL惩罚的参考策略
    """


    parser = argparse.ArgumentParser(
        description="MokioMind PPO (Proximal Policy Optimization)"
    )



    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "out"), help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="ppo_actor", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size（PPO batch较小）"
    )



    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument(
        "--critic_learning_rate", type=float, default=8e-8, help="Critic学习率"
    )


    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")


    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")


    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )



    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")


    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "rlaif-mini.jsonl"),
        help="RLAIF数据路径",
    )



    parser.add_argument(
        "--clip_epsilon",
        type=float,
        default=0.1,
        help="PPO裁剪参数（控制策略更新幅度）",
    )
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")



    parser.add_argument(
        "--reasoning",
        type=int,
        default=1,
        choices=[0, 1],
        help="推理模型类型（0=普通模型，1=推理模型）",
    )
    parser.add_argument(
        "--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率"
    )


    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "..", "internlm2-1_8b-reward"),
        help="Reward模型路径",
    )

    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )


    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-PPO", help="wandb项目名"
    )

    args = parser.parse_args()

    local_rank = init_distributed_mode()

    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))


    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR)
        if args.from_resume == 1
        else None
    )


    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )


    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MokioMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )


    base_weight = "reason" if args.reasoning == 1 else "full_sft"


    actor_model, tokenizer = init_model(lm_config, base_weight, save_dir=args.save_dir, device=args.device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    old_actor_model, _ = init_model(lm_config, base_weight, save_dir=args.save_dir, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)


    ref_model, _ = init_model(lm_config, base_weight, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)


    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)


    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )



    train_ds = RLHFDataset(
        args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len)
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(
        critic_model.parameters(), lr=args.critic_learning_rate
    )
    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, ((iters + args.accumulation_steps - 1) // args.accumulation_steps) * args.epochs)

    actor_scheduler = CosineAnnealingLR(
        actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    critic_scheduler = CosineAnnealingLR(
        critic_optimizer,
        T_max=total_optimizer_steps,
        eta_min=args.critic_learning_rate / 10,
    )


    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)


    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)


    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            ppo_train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                old_actor_model,
                ref_model,
                actor_scheduler,
                critic_scheduler,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
                )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            ppo_train_epoch(
                epoch,
                loader,
                len(loader),
                old_actor_model,
                ref_model,
                actor_scheduler,
                critic_scheduler,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )
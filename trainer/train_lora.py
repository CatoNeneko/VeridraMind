"""
MokioMind LoRA (Low-Rank Adaptation) 微调脚本  # ！修正：原MiniMind改为MokioMind

📚 LoRA 核心知识点：
- 什么是LoRA：一种参数高效微调方法，只训练少量新增参数
- 原理：在预训练模型的权重矩阵旁边添加低秩分解矩阵 ΔW = BA
  - 原始权重 W 保持冻结（requires_grad=False）
  - 新增两个小矩阵 A(d×r) 和 B(r×d)，其中 r<<d（秩远小于维度）
  - 前向计算：output = Wx + BAx
- 优势对比：
  - Full SFT：更新所有参数，效果好但需要大显存和长时间
  - LoRA：只更新1-5%的参数，显存需求小，训练快，适合资源受限场景
  - 多任务切换：可以保存多组LoRA权重，快速切换不同任务能力

📚 适用场景：
- 个性化定制：医疗、法律、金融等垂直领域适配
- 快速实验：尝试不同数据/超参时，LoRA训练速度快
- 资源受限：单卡或小显存环境
"""

import os
import sys




__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


from model.model import MokioMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)


warnings.filterwarnings("ignore")





def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行单个LoRA训练轮次

    📚 LoRA训练的特殊之处：
    1. 只有LoRA参数参与梯度计算和更新
    2. 原始模型权重保持冻结，节省显存和计算
    3. 训练流程与Full SFT相同，但参数量小得多

    Args:
        epoch: 当前训练轮次
        loader: 数据加载器
        iters: 总迭代次数
        lora_params: LoRA参数列表（只有这些参数会被更新）
        start_step: 起始步数（用于断点续训）
        wandb: 实验跟踪工具
    """
    start_time = time.time()



    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):




        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)



        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)




        for param_group in optimizer.param_groups:
            param_group["lr"] = lr




        with autocast_ctx:



            res = model(
                input_ids, labels=labels, attention_mask=attention_mask
            )



            loss = (
                res.loss + res.aux_loss
            )




            loss = loss / args.accumulation_steps





        scaler.scale(loss).backward()





        if step % args.accumulation_steps == 0 or step == iters:
            remainder = step % args.accumulation_steps
            if remainder:
                correction = args.accumulation_steps / remainder
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)



            scaler.unscale_(optimizer)






            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)





            scaler.step(optimizer)
            scaler.update()




            optimizer.zero_grad(set_to_none=True)



        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time



            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]





            eta_min = spend_time / max(1, step - start_step) * (iters - step) / 60




            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )


            if wandb:



                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )






        save_boundary = step % args.accumulation_steps == 0 or step == iters
        crossed_save_interval = step // args.save_interval > (step - args.accumulation_steps) // args.save_interval
        if (step == iters or (save_boundary and crossed_save_interval)) and is_main_process():
            model.eval()



            lora_save_path = (
                f"{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth"
            )





            save_lora(model, lora_save_path)






            lm_checkpoint(
                lm_config,
                weight=args.lora_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=CHECKPOINT_DIR,
            )

            model.train()




if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="MokioMind LoRA Fine-tuning")




    parser.add_argument(
        "--save_dir", type=str, default="../out/lora", help="模型保存目录"
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default="lora_identity",
        help="LoRA权重名称(如lora_identity/lora_medical等)",
    )





    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")




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




    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")






    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )





    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "lora_identity.jsonl"),
        help="LoRA训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="full_sft",
        type=str,
        help="基于哪个权重训练，默认full_sft",
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
        "--wandb_project", type=str, default="MokioMind-LoRA", help="wandb项目名"
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
        lm_checkpoint(lm_config, weight=args.lora_name, save_dir=CHECKPOINT_DIR)
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



        wandb_run_name = f"MokioMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )





    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=args.device)







    apply_lora(model)




    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(
        p.numel() for name, p in model.named_parameters() if "lora" in name
    )
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")




    lora_params = []


    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False






    train_ds = SFTDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)




    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None




    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))




    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)





    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)

        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)






    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])





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
            train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                lora_params,
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
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
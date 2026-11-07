import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import MokioMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    load_model_state,
    SkipBatchSampler,
)


warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个epoch的主核心函数

    Args:
        epoch: 当前epoch序号
        loader: 数据加载器
        iters: 该epoch总迭代次数
        start_step: 起始步数（用于断点续训）
        wandb: 实验跟踪系统
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





            res = model(input_ids, labels=labels, attention_mask=attention_mask)




            loss = res.loss + res.aux_loss





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


            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)


            scaler.step(optimizer)

            scaler.update()


            optimizer.zero_grad(set_to_none=True)



        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time

            current_loss = loss.item() * args.accumulation_steps


            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0

            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]


            eta_min = spend_time / max(1, step - start_step) * (iters - step) / 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )




        save_boundary = step % args.accumulation_steps == 0 or step == iters
        crossed_save_interval = (
            step // args.save_interval
            > (step - args.accumulation_steps) // args.save_interval
        )
        if (
            step == iters
            or (save_boundary and crossed_save_interval)
        ) and is_main_process():
            model.eval()


            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"




            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()



            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)




            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=checkpoint_dir,
                scaler=scaler,
            )

            model.train()
            del state_dict


        del input_ids, labels, attention_mask, res, loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MokioMind Full SFT")


    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="full_sft", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")


    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")


    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")


    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len",
        default=340,
        type=int,
        help="训练的最大截断长度（中文1token≈1.5~1.7字符）",
    )
    parser.add_argument(
        "--use_moe",
        default=1,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )


    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/sft_t2t_mini.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="pretrain",
        type=str,
        help="基于哪个权重训练，为none则不基于任何权重训练",
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
        "--wandb_project", type=str, default="MokioMind-Full-SFT", help="wandb项目名"
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用torch.compile加速（0=否，1=是）",
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_path == "../dataset/sft_t2t_mini.jsonl":
        args.data_path = os.path.join(project_root, "dataset", os.path.basename(args.data_path))
    elif not os.path.isabs(args.data_path) and not os.path.exists(args.data_path):
        args.data_path = os.path.join(project_root, args.data_path)
    if args.save_dir == "../out":
        args.save_dir = os.path.join(project_root, "out")
    checkpoint_dir = os.path.join(project_root, "checkpoints")



    """
    📚 分布式训练初始化知识点：
    - local_rank: 当前进程在本机上的GPU编号
    - 随机种子: 确保不同进程有不同但可复现的随机序列
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(
        42 + (dist.get_rank() if dist.is_initialized() else 0)
    )


    """
    📚 SFT特有：基于预训练模型微调
    - 通常from_weight='pretrain'，表示加载预训练权重
    - Pretrain脚本中from_weight='none'表示从头开始
    """
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=checkpoint_dir)
        if args.from_resume == 1
        else None
    )


    """
    📚 混合精度训练知识点：
    - bfloat16: Google开发，数值范围大，更稳定，推荐使用
    - float16: 标准半精度，节省内存但可能溢出
    - autocast: 自动选择精度，关键运算用float32
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )


    """
    📚 实验跟踪系统知识点：
    - SwanLab: 国产替代WandB的方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MokioMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )



    """
    📚 SFT vs Pretrain 数据集差异：
    - SFT: SFTDataset - 监督微调数据集，包含instruction和response
    - Pretrain: PretrainDataset - 预训练数据集，包含原始文本和mask
    """
    model, tokenizer = init_model(
        lm_config,
        "none" if ckp_data else args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
    )


    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")


    train_ds = SFTDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)


    """
    📚 断点续训恢复：
    - 模型参数状态
    - 优化器状态（动量、方差估计等）
    - 梯度缩放器状态
    - 训练进度（epoch和step）
    """
    start_epoch, start_step = 0, 0
    if ckp_data:
        load_model_state(model, ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)


    """
    📚 DistributedDataParallel特殊处理：
    - freqs_cos, freqs_sin是RoPE位置编码缓存，不需要梯度同步
    - 这样可以避免不必要的通信开销
    """
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])



    for epoch in range(start_epoch, args.epochs):

        train_sampler and train_sampler.set_epoch(epoch)


        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()



        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)


    if dist.is_initialized():
        dist.destroy_process_group()
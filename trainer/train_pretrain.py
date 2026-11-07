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
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
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



    start_time = time.time()


    for step, batch in enumerate(loader, start=start_step+1):







        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']




        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)


        lr = get_lr(epoch*iters+step, args.epochs*iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


        with autocast_ctx:




            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


            loss = res.loss + getattr(res, "aux_loss", 0.0)


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

            optimizer.zero_grad(set_to_none= True)







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
        crossed_save_interval = (
            step // args.save_interval
            > (step - args.accumulation_steps) // args.save_interval
        )
        if (
            step == iters
            or (save_boundary and crossed_save_interval)
        ) and is_main_process():
            model.eval()


            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"



            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()





            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)




            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=checkpoint_dir,
            )

            model.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MokioMind Pretraining")


    parser.add_argument(
        "--save_dir", type=str, default="../out", help="模型保存目录"
    )
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")


    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")


    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")


    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
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
        default="dataset/pretrain_t2t_mini.jsonl",
        help="预训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
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
        "--wandb_project", type=str, default="MokioMind-Pretrain", help="wandb项目名"
    )


    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_path == "dataset/pretrain_t2t_mini.jsonl":
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
    - 这样既保证了随机性，又保证了可复现性
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"




    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))


    """
    📚 模型配置和检查点管理：
    - 创建保存目录
    - 构建模型配置对象
    - 尝试加载断点续训数据
    """
    os.makedirs(args.save_dir, exist_ok=True)


    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )



    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir=checkpoint_dir
        )
        if args.from_resume == 1
        else None
    )


    """
    📚 混合精度训练知识点：
    - bfloat16: Google开发，数值范围大，更稳定
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
    - WandB: 实验管理平台，记录训练过程
    - SwanLab: 国产替代方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():

        import swanlab as wandb



        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None


        wandb_run_name = f"MokioMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )



    """
    📚 训练组件初始化：
    - 模型: 根据配置创建MiniMind模型
    - 数据集: 加载预训练数据
    - 采样器: 分布式训练的数据分配
    - 优化器: AdamW优化器
    - 缩放器: 混合精度训练的梯度缩放
    """

    model, tokenizer = init_model(
        lm_config,
        "none" if ckp_data else args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
    )

    train_ds = PretrainDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:

        load_model_state(model, ckp_data["model"])

        optimizer.load_state_dict(ckp_data["optimizer"])

        scaler.load_state_dict(ckp_data["scaler"])

        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():


        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])


    for epoch in range(start_epoch, args.epochs):


        if train_sampler:
            train_sampler.set_epoch(epoch)
            sample_order = train_sampler
        else:

            generator = torch.Generator().manual_seed(42 + epoch)
            sample_order = torch.randperm(len(train_ds), generator=generator).tolist()


        skip_batches = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(sample_order, args.batch_size, skip_batches)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if skip_batches > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{skip_batches}个step，从step {skip_batches + 1}开始"
            )
        train_epoch(epoch, loader, len(loader) + skip_batches, skip_batches, wandb)

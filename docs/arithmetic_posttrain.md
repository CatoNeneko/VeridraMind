# 整数题后训练

以下 PowerShell 命令从 VeridraMind 项目根目录运行，使用现有 `full_sft_512_moe.pth`。新权重前缀分别为 `arithmetic_sft` 和 `arithmetic_grpo`。

1. 生成固定种子的训练题和留出题。题库只包含 2 到 31 的整数加减乘法。留出集用于训练后的对照。

```powershell
.\.venv\Scripts\python.exe -B -m method.arithmetic_posttrain prepare --output-dir dataset/arithmetic
```

2. 每题采样四个回答。判题器要求整条回复是一个正确整数，每题最多保留一个通过的回答。

```powershell
.\.venv\Scripts\python.exe -B -m method.arithmetic_posttrain filter --input dataset/arithmetic/train.jsonl --output dataset/arithmetic/filtered.jsonl --weight full_sft --use-moe 1
```

3. 用筛选样本做一轮 SFT。如果采样通过率太低，先把下面的数据路径换成 `dataset/arithmetic/train.jsonl`，并把保存前缀换成 `arithmetic_warmup`。随后从热身权重重新筛选，正式筛选 SFT 也从热身权重开始。纯整数题的微调结果适合这个专项实验。

```powershell
.\.venv\Scripts\python.exe -B trainer/train_full_sft.py --data_path dataset/arithmetic/filtered.jsonl --from_weight full_sft --save_weight arithmetic_sft --use_moe 1 --epochs 1 --batch_size 8 --max_seq_len 128 --num_workers 0
```

4. 从 SFT 权重接入规则奖励 GRPO。GRPO 用完整训练题库出题，从题库的最后一条消息读取标准答案，只向策略传入题目。参考策略仍用于原有 KL 项。观察平均奖励和 mixed groups（组内既有正确又有错误回答的比例），以区分有无可用的组内奖励差。

```powershell
.\.venv\Scripts\python.exe -B trainer/train_grpo.py --reward_mode arithmetic --reasoning 0 --data_path dataset/arithmetic/train.jsonl --from_weight arithmetic_sft --save_weight arithmetic_grpo --use_moe 1 --epochs 1 --batch_size 2 --num_generations 4 --max_seq_len 128 --max_gen_len 16 --num_workers 0
```

需要比较两种方法时，让筛选 SFT 和 GRPO 分别从同一个底座权重出发，用同一份留出题和相同解码设置测准确率。顺序接入的命令展示的是 SFT 后继续 GRPO，回答的是继续训练能否改善这一专项。

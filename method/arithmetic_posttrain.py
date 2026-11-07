"""整数任务的出题、判题和筛选 SFT。运行 python -m method.arithmetic_posttrain --help。"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGER = re.compile(r"[+-]?[0-9]{1,12}", re.ASCII)


def is_correct(response, answer):
    """整条回复必须是一个整数，避免从多答案或解释中误取正确数字。"""
    response = response.strip()
    return INTEGER.fullmatch(response) is not None and int(response) == int(answer)


def arithmetic_rewards(answers, responses, generations):
    """generate 按题目排列候选，行号 i*G+j 对应第 i 道题的第 j 条回复。"""
    if generations < 1 or len(responses) != len(answers) * generations:
        raise ValueError("候选数必须等于题目数乘以每题采样数")
    return [
        float(is_correct(response, answers[i // generations]))
        for i, response in enumerate(responses)
    ]


def make_examples(train_count, holdout_count, seed):

    rows = []
    for op in ("+", "-", "*"):
        for a in range(2, 32):
            for b in range(2, 32):
                if op != "-" and a > b:
                    continue
                result = {"+": a + b, "-": a - b, "*": a * b}[op]
                rows.append({"conversations": [
                    {"role": "user", "content": f"计算 {a} {op} {b}，只输出一个整数。"},
                    {"role": "assistant", "content": str(result)},
                ]})
    if min(train_count, holdout_count) < 1 or train_count + holdout_count > len(rows):
        raise ValueError(f"训练和留出样本数须为正，合计最多 {len(rows)}")
    random.Random(seed).shuffle(rows)
    return rows[:train_count], rows[train_count:train_count + holdout_count]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_samples(args):
    import torch

    from model.model import MokioMindConfig
    from trainer.trainer_utils import init_model, setup_seed

    if min(args.generations, args.batch_size, args.max_new_tokens) < 1:
        raise ValueError("采样数、批大小和最大生成长度须为正")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("输入题库为空")
    setup_seed(args.seed)
    config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model, tokenizer = init_model(config, args.weight, save_dir=str(args.save_dir), device=args.device)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    accepted = []
    correct_count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    row["conversations"][:1], tokenize=False, add_generation_prompt=True
                )
                for row in batch
            ]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
            inputs = {key: inputs[key].to(args.device) for key in ("input_ids", "attention_mask")}
            width = inputs["input_ids"].shape[1]
            ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
                num_return_sequences=args.generations,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            replies = tokenizer.batch_decode(ids[:, width:], skip_special_tokens=True)
            answers = [row["conversations"][-1]["content"] for row in batch]
            scores = arithmetic_rewards(answers, replies, args.generations)
            correct_count += sum(scores)

            for i, row in enumerate(batch):
                first = i * args.generations
                for j in range(first, first + args.generations):
                    if scores[j]:
                        accepted.append({"conversations": [
                            row["conversations"][0],
                            {"role": "assistant", "content": replies[j].strip()},
                        ]})
                        break
    print(f"通过题目 {len(accepted)}/{len(rows)}，通过候选 {int(correct_count)}/{len(rows)*args.generations}")
    if not accepted:
        raise SystemExit("没有可用的筛选样本。先用训练题库做 SFT 热身，再重新采样。")
    write_jsonl(args.output, accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare", help="生成互斥的训练与留出题库")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--train-count", type=int, default=512)
    prepare.add_argument("--holdout-count", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=42)

    sample = modes.add_parser("filter", help="从已有权重采样，导出通过判题的 SFT 数据")
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--weight", default="full_sft")
    sample.add_argument("--save-dir", type=Path, default=ROOT / "out")
    sample.add_argument("--device", default="cuda:0")
    sample.add_argument("--hidden-size", type=int, default=512)
    sample.add_argument("--num-hidden-layers", type=int, default=8)
    sample.add_argument("--use-moe", type=int, choices=[0, 1], default=1)
    sample.add_argument("--batch-size", type=int, default=4)
    sample.add_argument("--generations", type=int, default=4)
    sample.add_argument("--max-new-tokens", type=int, default=16)
    sample.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "filter":
        filter_samples(args)
        return

    train, holdout = make_examples(args.train_count, args.holdout_count, args.seed)
    train_path = args.output_dir / "train.jsonl"
    holdout_path = args.output_dir / "holdout.jsonl"
    for path in (train_path, holdout_path):
        if path.exists():
            raise FileExistsError(path)
    write_jsonl(train_path, train)
    write_jsonl(holdout_path, holdout)
    print(f"已生成 {len(train)} 道训练题和 {len(holdout)} 道留出题：{args.output_dir}")


if __name__ == "__main__":
    main()

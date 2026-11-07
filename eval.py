import argparse
from pathlib import Path
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import load_model_state, setup_seed

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent



def init_model(args):



    local_path = (PROJECT_ROOT / Path(args.load_from).expanduser()).resolve()
    load_from = str(local_path) if local_path.is_dir() else args.load_from
    load_format = getattr(args, "load_format", "auto")
    native = load_format == "native" or (
        load_format == "auto" and local_path == (PROJECT_ROOT / "model").resolve()
    )
    if native:

        moe_suffix = "_moe" if args.use_moe else ""
        save_dir = PROJECT_ROOT / Path(args.save_dir).expanduser()
        ckp = save_dir / f"{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        if not ckp.is_file():
            raise FileNotFoundError(
                f"找不到权重：{ckp}。请核对 --save_dir、--weight、--hidden_size 和 --use_moe。"
            )
        tokenizer = AutoTokenizer.from_pretrained(load_from)

        model = MokioMindForCausalLM(
            MokioMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )

        load_model_state(model, torch.load(ckp, map_location="cpu", weights_only=True))
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_from)
        model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    print(f"VeridraMind 模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)")

    return model.eval().to(args.device), tokenizer



def main():
    parser = argparse.ArgumentParser(
        description="VeridraMind 模型推理与对话"
    )
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="tokenizer/模型目录或Hugging Face模型名，默认项目model目录",
    )
    parser.add_argument(
        "--load_format", choices=["auto", "native", "transformers"], default="auto",
        help="auto仅将本项目model目录作为原生.pth入口；其他目录可显式指定native",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="原生权重目录；相对路径以项目根为准")
    parser.add_argument(
        "--weight",
        default="pretrain",
        type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="保留参数：本脚本尚未实现LoRA加载，仅接受None",
    )

    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=8,
        type=int,
        help="隐藏层数量（Small/MoE=8, Base=16）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=False,
        action="store_true",
        help="启用YaRN位置编码外推（当前配置16倍，仅改变位置编码尺度）",
    )

    parser.add_argument(
        "--max_new_tokens",
        default=8192,
        type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="生成温度，须大于0；通常越大分布越平缓",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )

    parser.add_argument(
        "--historys",
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    args = parser.parse_args()
    if args.lora_weight != "None":
        parser.error("本脚本尚未实现LoRA加载，--lora_weight只能为None")

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]


    conversation = []

    model, tokenizer = init_model(args)

    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0:
            print(f"👶: {prompt}")

        conversation = conversation[-args.historys :] if args.historys else []
        conversation.append({"role": "user", "content": prompt})


        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if args.weight == "reason":
            templates["enable_thinking"] = True

        inputs = (
            tokenizer.apply_chat_template(**templates)
            if args.weight != "pretrain"
            else ((tokenizer.bos_token or "") + prompt)
        )


        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print("🤖️: ", end="")



        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )

        conversation.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()
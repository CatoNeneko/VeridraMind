# VeridraMind

VeridraMind 是一个小语言模型后训练项目，基于 [MiniMind 课程复现 Wood-Q/MokioMind](https://github.com/Wood-Q/MokioMind) 扩展。项目新增了独立的整数任务模块，涵盖题库生成、规则判题、采样筛选 SFT，以及可切换的 GRPO 可验证奖励。训练命令见[整数题后训练](arithmetic_posttrain.md)。

## 代码阅读路线

这份代码用于对照课程理解实现。先沿一条数据链路读通，再横向比较各训练方法，避免一开始就在命令行参数和分布式包装中迷路。

## 先统一形状

- `B`：一个微批次的样本数；`S`：训练输入长度；`C`：隐藏宽度；`V`：词表大小。
- 模型内部注意力：`Hq` 为查询头数，`Hkv` 为KV头数，`D=C/Hq`；KV缓存的历史长度记作 `P`。
- RL训练脚本：`P` 表示补齐后的prompt宽度，`R` 表示生成回复长度，`L=P+R`；GRPO每个prompt采样 `G` 个回答，展开后批量为 `B*G`。
- 符号以各文件开头/函数旁定义为准；Python列表、字符串、配置对象和整数不是张量。

## 建议顺序

| 顺序 | 文件 | 先抓住的主线 |
|---|---|---|
| 1 | `model/model.py` | RMSNorm → RoPE/YaRN → GQA → SwiGLU/MoE → Block → 主干 → LM输出 |
| 2 | `dataset/lm_dataset.py` | 原始文本/对话 → 模板 → 编号 → attention mask与监督标签 |
| 3 | `trainer/train_pretrain.py` | `[B,S]`输入 → `[B,S,V]`分数 → next-token CE → 参数更新 |
| 4 | `trainer/train_full_sft.py` | 模型结构相同，只在assistant回答位置提供监督 |
| 5 | `model/model_lora.py`、`trainer/train_lora.py` | 冻结底座，给选定线性层增加低秩可训练分支 |
| 6 | `trainer/train_dpo.py` | chosen/rejected配对 → 策略/参考log概率 → 偏好损失 |
| 7 | `trainer/train_ppo.py` | 在线采样 → 奖励 → Critic优势 → 策略裁剪与价值回归 |
| 8 | `trainer/train_grpo.py` | 每题多次采样 → 组内相对优势 → token策略梯度和KL项 |
| 9 | `trainer/trainer_utils.py` | 学习率、种子、分布式、权重加载、检查点和跳过批次 |
| 10 | `eval.py` | 重建模型/加载权重 → 模板 → 生成 → 切掉输入再解码 |

`method/moe.py`是实现正确性自检，不是训练效果评测；不会返回语言能力评分。`method/rmsnorm.py`是小演示，`method/rope.py`尚未实现独立演示；实际模型实现都在`model/model.py`。

## 读一段代码时问四件事

1. 这一行接收的是字符串、token编号还是隐藏特征？shape是什么？
2. 运算改变了哪些轴，哪些轴只是被广播或重新排列？
3. 结果用于生成、监督、奖励还是日志？是否参与反向传播？
4. 哪些参数会拿到梯度，真正的optimizer.step在哪里发生？

## 几个容易混淆的接口

- 预训练/SFT/LoRA的`labels`与输入仍按原序列对齐，模型内部再shift；DPO数据集已构造`x=[:-1]`、`y=[1:]`，训练器不能重复shift。
- `attention_mask=0`控制padding的可见性；`labels=-100`或DPO的`loss mask=0`控制哪些目标计入损失。SFT的用户问题不算监督，但模型仍然需要看见它。
- `model.eval()`只改变Dropout等层的行为，不会自动禁用梯度；`no_grad()`禁止记录运算图，`requires_grad_(False)`冻结参数，三者作用不同。
- `backward()`累加梯度，`step()`更新参数，`zero_grad()`清除梯度；三者不是同一件事。
- 普通权重文件用于推理或初始化；`*_resume.pth`还带优化器等状态。这里没有保存全部随机数和worker状态，不保证逐位复现原轨迹。

## 与理论公式对照时的边界

DPO沿用教程的**回复token平均log概率**，与使用序列log概率求和的标准DPO目标有差别。PPO采用**序列级优势与序列级概率比值**的简化实现，没有完整的逐token GAE、多轮mini-batch PPO更新；日志中的采样log概率差也不是精确的全分布KL。GRPO沿用教程的单次在线更新和组内优势处理。这些差异已在对应代码中注明，阅读时不要把教学实现当作所有论文/框架的唯一实现。

真实训练依赖匹配的tokenizer、底座权重、数据和奖励模型。小型CPU检查验证接口和计算边界，不能证明真实训练收敛或模型能力提升。

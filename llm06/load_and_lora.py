# load_and_lora.py
# 4.1.3 加载基座模型与配置 LoRA
# 用途: 加载 Qwen2.5-0.5B-Instruct 基座, 冻结全部原始权重,
#       只注入 LoRA 低秩适配层（r=8）——训练参数量 <1%。
# 关键参数:
#   LoRA r=8, alpha=16      秩与缩放; r 越大容量越大、显存/算力开销越高
#   target_modules          对注意力层的 q/k/v/o 投影注入适配器
#   task_type="CAUSAL_LM"   因果语言模型任务
# 预期输出: 打印可训练参数量与占比(通常 <1%), 以及 LoRA 注入后的模型结构摘要。
# 常见报错与解决:
#   - CUDA 不可用 -> 装的是 CPU 版 torch, 需重装 cu126 版;
#   - target_modules 名字不匹配 -> 不同模型模块名不同, Qwen2 为 q/k/v/o_proj;
#   - OOM -> 减小 r 或只对 q_proj/v_proj 注入。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# —— 本机有 RTX 4050 (6GB): 按教程方案 1 半精度加载, 自动分配设备 ——
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # 半精度省显存
    device_map="auto",           # 自动分配设备(GPU)
)

lora_cfg = LoraConfig(
    r=8,                          # LoRA 秩
    lora_alpha=16,                # 缩放系数 (常取 2*r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()   # 应显示 trainable < 1%

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("LoRA 注入完成, 模型就绪。")

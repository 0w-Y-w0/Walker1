# format_data.py
# 4.1.2 数据格式化（拼接为对话/指令文本并分词）
# 用途: 把 train.json 的 instruction+input 拼成"提问"，output 作为"回答"，
#       用 Qwen2.5-0.5B-Instruct 的 tokenizer 分词，并构造 labels:
#       仅输出部分计算 loss（-100 屏蔽 prompt 部分, 标准 SFT 做法）。
# 关键参数:
#   MODEL_NAME   基座模型, 0.5B 参数 CPU 可训
#   截断长度 512  统一训练长度
# 预期输出: 打印 tok_ds[0]（input_ids/labels/attention_mask）。
# 常见报错与解决:
#   - 下载模型超时 -> 已在文件头设置 HF_ENDPOINT=hf-mirror.com 国内镜像;
#   - OOM -> 减小截断长度(256)或 batch_size。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import json
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 1) 读入数据
with open("train.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
ds = Dataset.from_list(raw)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Qwen 系列若没有 pad_token, 用 eos 充当
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) 把 instruction+input 拼成"提问"，output 作为"回答"
def build_prompt(e):
    return (
        f"{e['instruction']}\n\n"
        f"工单内容: {e['input']}\n\n"
        f"请输出决策: "
    )

def tokenize_function(e):
    prompt = build_prompt(e)
    # 输入部分不计算损失，仅对输出部分算 loss（标准 SFT 做法）
    full = prompt + e["output"] + tokenizer.eos_token
    # 分别编码，便于后续用 labels 屏蔽 prompt
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(e["output"] + tokenizer.eos_token,
                      add_special_tokens=False)["input_ids"]
    input_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids  # -100 在 cross_entropy 中忽略
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}

tok_ds = ds.map(tokenize_function, remove_columns=ds.column_names,
                batched=False)
# 为训练统一长度
tok_ds = tok_ds.map(lambda x: {"input_ids": x["input_ids"][:512],
                               "labels": x["labels"][:512],
                               "attention_mask": x["attention_mask"][:512]},
                    batched=False)
print(tok_ds[0])

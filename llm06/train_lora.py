# train_lora.py
# 4.1.4 训练参数与 Trainer
# 用途: 用 HF Trainer 在 Qwen2.5-0.5B + LoRA 上做 SFT 微调，
#       只训练 0.22% 的适配器参数，训练后仅保存 LoRA 权重。
# 关键参数:
#   per_device_train_batch_size=2 显存小就设 1 (6GB 跑 0.5B 够用)
#   gradient_accumulation_steps=4 梯度累积, 模拟更大 batch
#   num_train_epochs=3           小数据集跑几轮
#   learning_rate=2e-4           LoRA 常用较大学习率
#   fp16=True                    GPU 半精度 (本机 RTX 4050 可用)
#   save_strategy="epoch" / report_to="none"  按 epoch 存档、不接 wandb
# 预期输出: 训练日志(loss 逐下降), 结束后在 ./qwen05-lora-ticket/adapters
#           保存 LoRA 适配器(仅几十 MB)。
# 常见报错与解决:
#   - OOM -> per_device_train_batch_size=1 或去掉 fp16;
#   - fp16 警告 -> GPU 不支持半精度时改 bf16=True 或删掉该行(CPU 必须);
#   - loss 不降 -> 数据太少属正常, 观察过拟合需留出验证集。
from load_and_lora import model, tokenizer   # 基座 + LoRA 注入
from format_data import tok_ds               # 已分词训练集

from transformers import TrainingArguments, Trainer
import torch

# 自定义 collator: 动态 pad 到同批最长; labels 用 -100 pad(不计 loss)
# 注: DataCollatorForLanguageModeling 对预分词且含 -100 的 labels 会报
#     "Unable to create tensor", 因此手写。
def collator(features):
    maxlen = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        pad = maxlen - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        attn.append(f["attention_mask"] + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}

training_args = TrainingArguments(
    output_dir="./qwen05-lora-ticket",
    per_device_train_batch_size=2,      # 显存小就设 1
    gradient_accumulation_steps=4,      # 模拟更大 batch
    num_train_epochs=3,                 # 小数据集多跑几轮
    learning_rate=2e-4,                 # LoRA 常用较大学习率
    fp16=True,                          # GPU 半精度; CPU 去掉
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",                   # 不接 wandb
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("./qwen05-lora-ticket/adapters")  # 仅保存 LoRA 适配器
print("LoRA 适配器已保存到 ./qwen05-lora-ticket/adapters")

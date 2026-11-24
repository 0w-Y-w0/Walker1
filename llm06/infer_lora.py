# infer_lora.py
# 4.1.5 用微调后的适配器做决策推理
# 用途: 基座模型 + LoRA 适配器合成"微调后模型", 对新工单做
#       类别判断与处理建议——验收微调效果。
# 关键参数:
#   max_new_tokens=128  生成长度上限
#   do_sample=False, temperature=1.0  贪心解码, 输出稳定可复现
#   PeftModel.from_pretrained(base, ADAPTER)  把 4.2MB 适配器叠加到基座上
# 预期输出: 微调后模型按 "类别: ...\n处理建议: ..." 的训练格式作答。
# 常见报错与解决:
#   - 输出重复/胡话 -> 数据太少属正常; 增加样本或增大 epochs 可改善;
#   - CUDA OOM -> device_map="cpu" 或减小 max_new_tokens;
#   - 提示词必须与训练时 build_prompt 完全一致, 否则模型"不认识"任务。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "./qwen05-lora-ticket/adapters"

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def decide(ticket_text):
    prompt = (
        "你是客服工单智能决策助手。请判断工单类别，并给出可执行的处理建议。\n\n"
        f"工单内容: {ticket_text}\n\n请输出决策: "
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             temperature=1.0)
        return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":   # 直接运行时自检; 被 decision_pipeline 导入时不执行
    print(decide("用户投诉: 刚买的空调能退款吗？用了 2 天坏了。"))

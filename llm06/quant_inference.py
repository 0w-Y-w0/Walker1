# quant_inference.py
# 4.3.1 4bit/8bit 量化与 transformers 内量化（最易上手）
# 用途: 用 BitsAndBytesConfig 做 NF4 4bit 量化加载, 显存降到约 1/4;
#       这是 QLoRA 推理侧的标配。
# 关键参数:
#   load_in_4bit=True            4bit 量化开关
#   bnb_4bit_compute_dtype=fp16  反量化计算精度
#   bnb_4bit_quant_type="nf4"    NormalFloat4, 质量优于普通 int4
#   bnb_4bit_use_double_quant=True 二次量化, 再省约 0.4 bit/参数
# 预期输出: 打印 4bit 加载后的显存占用, 与 fp16 加载对比显著下降。
# 常见报错与解决:
#   - 需要 CUDA GPU: bitsandbytes 量化仅支持 NVIDIA 卡 (本机 RTX 4050 OK);
#   - 'bitsandbytes' 未装 -> pip install bitsandbytes>=0.41;
#   - 0.5B 模型本来就小, 量化收益在 7B+ 模型上更明显。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4bit 量化加载（QLoRA 推理侧）: 显存降到约 1/4
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             quantization_config=bnb,
                                             device_map="auto")

print("4bit 量化模型已加载，显存占用显著降低。")
print("当前 GPU 显存占用: %.1f MB"
      % (torch.cuda.memory_allocated() / 1024 / 1024))

# 快速自检: 量化模型仍能正常生成
prompt = "请用一句话说明什么是模型量化。"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print("自检输出:", tok.decode(out[0], skip_special_tokens=True))

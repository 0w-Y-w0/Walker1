# app.py
# 4.2 实验B: 本地部署推理服务 (FastAPI + /generate)
# 用途: 把（微调后的）模型封装成 HTTP 服务, 供其它系统以 JSON 调用,
#       形成可嵌入决策系统的模块。
# 接口:
#   POST /generate  请求体 {"prompt":..., "max_new_tokens":128, "do_sample":false}
#                   返回 {"text": "..."}
#   GET  /health    健康检查, 返回 {"status": "ok"}
# 启动: uvicorn app:app --port 8000   (在 llm06 目录下)
# 常见报错与解决:
#   - 首次启动慢: 模型加载到显存需数十秒, 属正常;
#   - PeftModel 路径不存在报错: 把 ADAPTER 改成不存在以走 except 用基座;
#   - 端口被占用: 换 --port 8001。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "./qwen05-lora-ticket/adapters"  # 没有微调结果可注释掉 PeftModel 那行

app = FastAPI(title="智能决策大模型服务", version="0.1")

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                            device_map="auto")
try:
    model = PeftModel.from_pretrained(base, ADAPTER)
except Exception:
    model = base  # 未微调时直接用基座
model.eval()


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    do_sample: bool = False


@app.post("/generate")
def generate(req: GenReq):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens,
                             do_sample=req.do_sample)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return {"text": text}


@app.get("/health")
def health():
    return {"status": "ok"}

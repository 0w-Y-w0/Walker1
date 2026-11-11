# step1_client.py
# 4.1 建立统一的客户端（云端 / 本地一键切换）
# 用途: 根据是否配置云端 Key，自动选择"云端 API"或"本地 Ollama"，
#       后续所有调用复用同一函数——上层代码不感知后端差异。
# 关键参数:
#   api_key / base_url / model 来自环境变量；若缺失则自动回落到本地 Ollama。
#   LOCAL_BASE / LOCAL_MODEL 为 Ollama 本地服务地址与模型名。
# 预期输出: 打印当前使用哪个后端；返回可用的 OpenAI 客户端与模型名。
# 常见报错与解决:
#   - AuthenticationError -> 云端 Key 错误或 base_url 拼错；检查环境变量。
#   - ConnectionError 连 localhost:11434 失败 -> Ollama 未启动:
#     先 `ollama run qwen2.5:7b` 保活。
import os
from openai import OpenAI

CLOUD_KEY = os.getenv("OPENAI_API_KEY")
CLOUD_BASE = os.getenv("OPENAI_BASE_URL")
CLOUD_MODEL = os.getenv("OPENAI_MODEL")

LOCAL_BASE = "http://localhost:11434/v1"
LOCAL_MODEL = "qwen2.5:7b"

if CLOUD_KEY:
    client = OpenAI(api_key=CLOUD_KEY, base_url=CLOUD_BASE)
    model = CLOUD_MODEL
    backend = "云端 API（%s）" % CLOUD_BASE
else:
    client = OpenAI(api_key="ollama", base_url=LOCAL_BASE)
    model = LOCAL_MODEL
    backend = "本地 Ollama（%s）" % LOCAL_BASE

print("当前后端:", backend, "| 模型:", model)

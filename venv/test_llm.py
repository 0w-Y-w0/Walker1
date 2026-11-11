# test_llm.py
# 用途: 验证 DeepSeek API 连通性——确认环境变量里的 Key/BaseURL/Model 可用。
# 说明: 本脚本不打印密钥，只打印模型回复与 token 用量。
# 常见报错与解决:
#   - AuthenticationError(401) -> Key 错误或已被删除，重新申请;
#   - APIConnectionError -> 网络/代理问题，检查能否访问 api.deepseek.com;
#   - 取到 None -> 环境变量只在"新开的终端"里生效，需重启 VS Code 终端。
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
model = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not api_key:
    raise SystemExit("未读取到 OPENAI_API_KEY，请重启终端后再试。")

print("Base URL:", base_url)
print("Model   :", model)
print("Key     :", "已读取（长度 %d，前缀 %s...）" % (len(api_key), api_key[:5]))

client = OpenAI(api_key=api_key, base_url=base_url)

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是一个严谨的中文助教，回答简洁。"},
        {"role": "user", "content": "用一句话说明什么是类别不平衡。"},
    ],
    temperature=0.7,
    max_tokens=200,
)

print("\n模型回复:", resp.choices[0].message.content)
print("token 用量:", resp.usage.prompt_tokens, "输入 /",
      resp.usage.completion_tokens, "输出")

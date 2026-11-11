# step5_json_decision.py
# 4.5 结构化输出: JSON 模式（决策自动化）
# 用途: 构造决策问答，要求模型输出结构化 JSON
#       （是否补货、建议补货量、理由），便于下游系统直接读。
# 关键参数:
#   response_format={"type":"json_object"} 强制 JSON 输出
#     （OpenAI 兼容云端支持; Ollama 用 extra_body={"format":"json"}）
#   temperature=0.0 决策需稳定
# 预期输出: 打印一段合法 JSON 字符串, 含
#   should_replenish / suggested_qty / reason 字段。
# 常见报错与解决:
#   - InvalidRequestError: json_object 不支持
#       -> 该后端不支持 response_format; 改用提示词强制"只输出 JSON"
#          并 json.loads 容错解析。
#   - JSONDecodeError -> 模型偶尔在 JSON 前后加多余文字;
#       用正则抽取第一个 {...} 再解析。
import json, re
from step1_client import client, model, CLOUD_KEY, CLOUD_BASE

system = (
    "你是供应链决策引擎。仅输出 JSON，不要任何解释文字。"
    "字段: should_replenish(布尔), suggested_qty(整数,件), reason(字符串)。"
    "规则: 若 库存 < 日均销量*提前期, 则应补货, 建议量=日均销量*提前期*1.5 - 库存。"
)

user = "当前库存 120。日均销量 40。补货提前期 5 天。"

kwargs = {"model": model, "messages": [
    {"role": "system", "content": system},
    {"role": "user", "content": user}],
    "temperature": 0.0, "max_tokens": 300}

# 云端用 response_format; 本地 Ollama 用 format
if "localhost" in (CLOUD_BASE or "") or (not CLOUD_KEY):
    kwargs["extra_body"] = {"format": "json"}
else:
    kwargs["response_format"] = {"type": "json_object"}

r = client.chat.completions.create(**kwargs)
raw = r.choices[0].message.content
print("原始返回:\n", raw)

# 容错解析: 抽取第一个 {...} 块
m = re.search(r"\{.*\}", raw, re.DOTALL)
decision = json.loads(m.group(0)) if m else None
print("\n解析后决策:", decision)

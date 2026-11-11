# step2_first_call.py
# 4.2 第一次调用——让模型做一句决策解释
# 用途: 用最简单的 Zero-shot 调用，验证"客户端→模型→回答"链路畅通。
# 关键参数:
#   model          由步骤 1 决定（云端或本地）
#   messages       对话历史: system 设定角色, user 给任务
#   temperature=0.2 低温度，输出更稳定可复现
#   max_tokens=300 控制长度与成本
# 预期输出: 模型返回一段中文，对"是否该补货"给出简短理由。
# 常见报错与解决:
#   - RateLimitError -> 触发限流: 云端免费档有 QPS 限制，稍等重试或降低调用频率。
#   - 超时 -> 网络慢或本地模型首次加载慢: 本地可多等几秒。
from step1_client import client, model

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是一名严谨的供应链决策分析师，回答简洁、基于数据。"},
        {"role": "user", "content": "当前库存 120 件，日均销量 40 件，补货提前期 5 天。是否应该补货？为什么？"},
    ],
    temperature=0.2,
    max_tokens=300,
)
print(resp.choices[0].message.content)

# decision_pipeline.py
# 4.5 实验E: 把微调模型用于具体决策（综合落地）
# 用途: 复用实验 A 的微调模型, 演示批量工单决策与简单分流——
#       体会"通用大模型 + 领域微调 = 可用的智能决策模块"。
# 关键参数:
#   tickets          待处理工单列表
#   decide()         来自 infer_lora(基座+LoRA 适配器, 贪心解码)
#   ROUTE_RULES      按"类别"关键词把决策归入业务队列(模拟自动派单)
# 预期输出: 逐条打印工单与决策建议; 结尾打印派单结果
#           (售后/账务/技术 队列, 可解析后写库)。
# 常见报错与解决:
#   - import infer_lora 循环/路径问题: 确保在 llm06 目录下运行,
#     且 qwen05-lora-ticket/adapters 存在;
#   - 0.5B 模型输出偶有模板偏差, 派单匹配不到时落入"人工复核"队列兜底。
from infer_lora import decide

tickets = [
    "用户投诉: 同一笔订单被扣了两次款，要求退款。",
    "用户咨询: 会员年费怎么开发票？",
    "用户反馈: App 登录后一直转圈。",
]

# 简单分流规则: 决策文本包含关键词 -> 派入对应队列(后续可写入数据库)
ROUTE_RULES = {
    "售后": "售后队列",
    "退换": "售后队列",
    "发票": "账务队列",
    "退款": "账务队列",
    "技术": "技术队列",
}

results = []
for t in tickets:
    result = decide(t)
    print("=" * 40)
    print("工单:", t)
    print("决策:\n", result)
    # 解析类别并派单
    route = "人工复核"          # 兜底队列
    for kw, queue in ROUTE_RULES.items():
        if kw in result:
            route = queue
            break
    results.append((t, route))
    print("派单:", route)

print("=" * 40)
print("派单汇总(可写入数据库实现自动派单):")
for t, route in results:
    print(f"  [{route}] {t}")

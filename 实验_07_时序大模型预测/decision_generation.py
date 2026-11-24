# decision_generation.py
# 5.3 决策生成: 预测 -> 业务决策的闭环
# 用途: 把模型输出套入业务阈值与成本约束, 生成可执行建议,
#       而非只给一串数字——从"预测精度"走向"决策价值"。
# 关键参数: capacity(容量阈值)、current_stock、safety(安全库存)。
# 预期输出: 调度告警表(超过容量 90% 的小时) + 库存补货建议量 Q。
# 常见报错与解决:
#   - KeyError 'preds' -> 确认 evaluate_eval.py 已在同目录运行成功
#     (本脚本直接导入其变量, 保证决策用的是真实预测值);
#   - 告警为空: 说明测试首日无超容时段, 可调低 capacity 观察;
#   - 建议 Q 为负: 现库存充足, 无需补货(max(0,...)已兜底)。
import numpy as np
import pandas as pd
from evaluate_eval import preds, gts, pred_len

# --- 场景1: 电力负荷调度 ---
capacity = 80.0                  # 区域供电容量(示例值，与原量纲一致)
load_pred = preds[0, :, 0]       # 取第一条测试样本通道 0 的未来预测
schedule = pd.DataFrame({
    "未来小时": range(1, pred_len + 1),
    "预测负荷": np.round(load_pred, 2),
    "是否告警": ["是" if v >= 0.9 * capacity else "否" for v in load_pred],
})
print("=== 负荷调度建议（超过容量 90% 触发告警）===")
print(schedule[schedule["是否告警"] == "是"].to_string(index=False))
if schedule["是否告警"].eq("是").sum() == 0:
    print("（未来 24 小时均在容量 90% 以内, 无需告警）")

# --- 场景2: 销量库存补货 ---
current_stock = 40.0             # 当前库存
safety = 10.0                    # 安全库存
sales_pred = preds[0, :, 1]      # 通道 1 的未来销量预测
demand = sales_pred.sum()        # 未来 H 步总需求
reorder = max(0.0, demand + safety - current_stock)
print("\n=== 库存补货建议 ===")
print(f"未来{pred_len}步预测总需求: {demand:.2f}")
print(f"建议补货量 Q = max(0, 需求+安全库存-现有库存) = {reorder:.2f}")

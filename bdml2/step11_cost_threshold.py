# step11_cost_threshold.py
# 4.11 代价敏感决策与阈值选择（把概率变成动作）
# 用途: 本实验"智能决策"的灵魂——模型给出概率 p，业务需要决定"是否拦截"。
#       用代价矩阵在给定"漏掉代价 vs 误拦代价"下，搜索总期代价最小的阈值，
#       体现分类→决策。
# 关键参数:
#   C_FN=1000  漏掉一笔欺诈损失 1000 元（资金损失+声誉）
#   C_FP=50    误拦一笔正常交易损失 50 元（客户体验/客诉）
#   thresholds 在 0.01~0.99 扫描
# 预期输出: 打印最优阈值与对应 Recall/Precision; 并画出"总代价 vs 阈值"曲线，呈 U 形。
# 常见报错与解决:
#   - 曲线无最低点 -> 代价比例不合理，调整 C_FN/C_FP;
#   - 最优阈值=0.99 -> 误拦代价过高，模型几乎不拦截，需重设业务权重。
# 补充: 理论最优阈值公式 tau* = C_FN / (C_FN + C_FP)
#       （本例 = 1000/1050 ≈ 0.952，经验扫描结果应在其附近）。
import os
import numpy as np
from step1_config import *
from step4_split import y_test
from step6_rf import y_prob_rf

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- 1) 代价矩阵与阈值网格 ----------
C_FN = 1000.0   # 漏掉代价
C_FP = 50.0     # 误拦代价

thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
total_costs = []
recall_at = []

# ---------- 2) 逐阈值扫描: 统计混淆矩阵四格 -> 总代价 ----------
for tau in thresholds:
    pred = (y_prob_rf >= tau).astype(int)
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tp = int(((pred == 1) & (y_test == 1)).sum())
    cost = fp * C_FP + fn * C_FN
    total_costs.append(cost)
    recall_at.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

# ---------- 3) 找 U 形曲线最低点 ----------
best_idx = int(np.argmin(total_costs))
best_tau = thresholds[best_idx]
print(f"最优阈值 τ* = {best_tau:.2f} | 总期望代价最低 = {total_costs[best_idx]:.0f} 元")
print(f"  对应 Recall={recall_score(y_test, (y_prob_rf>=best_tau).astype(int)):.3f}, "
      f"Precision={precision_score(y_test, (y_prob_rf>=best_tau).astype(int)):.3f}")

# 理论最优阈值（贝叶斯决策推导: (1-tau)/tau = C_FP/C_FN）
tau_formula = C_FN / (C_FN + C_FP)
print(f"理论阈值公式 τ* = C_FN/(C_FN+C_FP) = {tau_formula:.3f}")

# ---------- 4) 可视化: 总代价 vs 阈值（U 形） ----------
plt.figure(figsize=(7, 5))
plt.plot(thresholds, total_costs, color="#2ca02c")
plt.axvline(best_tau, color="red", ls="--", label=f"最优阈值 τ*={best_tau:.2f}")
plt.axhline(total_costs[best_idx], color="gray", ls=":", alpha=0.6)
plt.xlabel("决策阈值 τ（拦截概率下限）")
plt.ylabel("总期望代价（元）")
plt.title("代价敏感: 总期望代价 vs 阈值（U 形，找最低点）")
plt.legend(); plt.tight_layout()
plt.savefig("fig_cost_threshold.png", dpi=110)
plt.show()

# ---------- 5) 默认 0.5 阈值 vs 最优阈值 的代价对比 ----------
for name, tau in [("默认阈值 0.5", 0.5), (f"最优阈值 {best_tau:.2f}", best_tau)]:
    pred = (y_prob_rf >= tau).astype(int)
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    print(f"  {name}: FN={fn} 笔 ×1000 + FP={fp} 笔 ×50 = 总代价 {fp*C_FP + fn*C_FN} 元")

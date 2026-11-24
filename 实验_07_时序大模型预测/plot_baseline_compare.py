# plot_baseline_compare.py
# 5.2 Baseline vs 深度时序模型对比
# 用途: 把 ARIMA / Prophet 与 PatchTST 在同一测试段上的 MAE/RMSE 并列比较。
# 关键参数: 对比指标 MAE、RMSE。
# 预期输出: 对比表 + figures/compare_baseline.png 柱状图。
#   教程为三根柱(含 Prophet); 本机按教程注释跳过 Prophet(Python 3.13 无
#   wheel), 仅对比 PatchTST vs ARIMA 两根柱, 报告中注明此差异。
#   深度模型取步骤8 evaluate() 的全测试段口径, ARIMA 取步骤8 的
#   同段对照口径(单通道、测试段前 pred_len 步), 报告需说明口径差异。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from evaluate_eval import mae, rmse, mae_arima, rmse_arima  # 真实计算结果

labels = ["PatchTST（深度）", "ARIMA"]
maes   = [mae,        mae_arima]
rmses  = [rmse,       rmse_arima]
print("对比表(原始量纲):")
print(f"{'模型':<16}{'MAE':>10}{'RMSE':>10}")
for l, a, b in zip(labels, maes, rmses):
    print(f"{l:<16}{a:>10.3f}{b:>10.3f}")

x = np.arange(len(labels)); w = 0.35
plt.figure(figsize=(8, 4))
plt.bar(x - w/2, maes,  w, label="MAE",  color="steelblue")
plt.bar(x + w/2, rmses, w, label="RMSE", color="darkorange")
plt.xticks(x, labels)
plt.ylabel("误差（越低越好）")
plt.title("Baseline vs 深度时序模型（测试段）")
plt.legend(); plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/compare_baseline.png", dpi=120)
print("已保存 figures/compare_baseline.png")

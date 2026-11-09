# step8_compare.py
# 用途: 在同一图上比较三类模型的 ROC 曲线与 PR 曲线，直观看出"AUC 高≠业务好"。
# 关键参数: 比较 y_prob_* 即可。
# 预期输出: 两张图，左 ROC（越靠左上越好），右 PR（不平衡时比 ROC 更敏感）。
# 常见报错与解决:
#   - 曲线重叠看不清 -> 拉大 figsize，或只画两个模型对比;
#   - PR 曲线往右下移动 -> 样本多是正常。
from step4_split import y_test
from step5_lr import y_prob_lr
from step6_rf import y_prob_rf
from step7_svm import y_prob_svm
from step1_config import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

models = {
    "LogReg": y_prob_lr,
    "RandomForest": y_prob_rf,
    "SVM": y_prob_svm,
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, prob in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("召回率 TPR")

axes[0].set_title("ROC 曲线对比"); axes[0].legend()

for name, prob in models.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    axes[1].plot(rec, prec, label=name)
axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
axes[1].set_title("PR 曲线对比（不平衡时更敏感）"); axes[1].legend()

plt.tight_layout(); plt.savefig("fig_roc_pr.png", dpi=120)
print("已保存 fig_roc_pr.png")

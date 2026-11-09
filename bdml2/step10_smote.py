# step10_smote.py
# 4.10 类别不平衡进阶处理（class_weight / 重采样 / SMOTE）
# 用途: 演示 SMOTE 过采样在训练集上生成合成少数类样本，提升召回率；
#       并对比"原始 vs SMOTE"的 F1（注意: SMOTE 只作用于训练集！）。
# 关键参数:
#   SMOTE(k_neighbors=5, sampling_strategy=0.2)
#     把正类过采样到负类的 20%（避免过度生成）
#   random_state=42 可复现（本章统一改用 RANDOM_STATE=482）
# 预期输出: SMOTE 后训练集正类数量明显增多；测试集 Recall 上升、
#           Precision 略降，F1 通常改善。
# 常见报错与解决:
#   - ValueError: Expected n_neighbors <= n_samples
#       -> k_neighbors 大于少数类样本数，调小 k_neighbors;
#   - 过拟合严重 -> 降低 sampling_strategy（如 0.1）
#     或在 SMOTE 后接欠采样 ENN。
# 小结三种不平衡处理思路:
#   1. class_weight（算法级）: 改动损失权重，最简单、首选;
#   2. 欠采样（数据级）: 减少负类，训练快但可能丢信息;
#   3. 过采样 / SMOTE（数据级）: 增加正类，改善召回但易过拟合，需配合正则。
import os
from step1_config import *
from step4_split import X_train_s, X_test_s, y_train, y_test
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- 1) 对训练集做 SMOTE 过采样（只碰训练集，测试集保持真实分布） ----------
sm = SMOTE(sampling_strategy=0.2, k_neighbors=5, random_state=RANDOM_STATE)
X_tr_sm, y_tr_sm = sm.fit_resample(X_train_s, y_train)
print("SMOTE 前训练集正类:", int(y_train.sum()),
      "-> 后:", int(y_tr_sm.sum()),
      "（正类占比 {:.2%} -> {:.2%}）".format(y_train.mean(), y_tr_sm.mean()))

# ---------- 2) 用过采样后的训练集训练随机森林 ----------
rf_sm = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                               random_state=RANDOM_STATE)
rf_sm.fit(X_tr_sm, y_tr_sm)
y_pred_sm = rf_sm.predict(X_test_s)

# ---------- 3) 在原始测试集上评估，并与 4.6 的 class_weight 版本对比 ----------
print("SMOTE+随机森林 测试集:")
print("  Accuracy :", round(accuracy_score(y_test, y_pred_sm), 4))
print("  Precision:", round(precision_score(y_test, y_pred_sm), 4))
print("  Recall   :", round(recall_score(y_test, y_pred_sm), 4))
print("  F1       :", round(f1_score(y_test, y_pred_sm), 4))

# 对照组: 4.6 中 class_weight="balanced_subsample" 的随机森林（不做重采样）
rf_base = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 random_state=RANDOM_STATE)
rf_base.fit(X_train_s, y_train)
y_pred_base = rf_base.predict(X_test_s)

print("\n对比: 原始(class_weight) vs SMOTE")
print("  原始  Precision/Recall/F1:",
      round(precision_score(y_test, y_pred_base), 4), "/",
      round(recall_score(y_test, y_pred_base), 4), "/",
      round(f1_score(y_test, y_pred_base), 4))
print("  SMOTE Precision/Recall/F1:",
      round(precision_score(y_test, y_pred_sm), 4), "/",
      round(recall_score(y_test, y_pred_sm), 4), "/",
      round(f1_score(y_test, y_pred_sm), 4))

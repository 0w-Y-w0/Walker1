# step6_rf.py
# 用途: 随机森林捕捉非线性与特征交互，通常在不平衡表格数据上强于逻辑回归。
# 关键参数:
#   n_estimators=300        树的数量，越多越稳（代价是训练时间）
#   class_weight="balanced_subsample"  每棵树自助采样后按分布加权
#   max_depth=None          不限制深度（可用 10~20 防止过拟合）
#   n_jobs=-1               并行训练，占满 CPU
# 预期输出: Recall 通常高于逻辑回归，F1 提升明显。
# 常见报错与解决:
#   - 训练慢 -> 减少 n_estimators 或设 max_depth;
#   - 内存占用高 -> 降低 n_estimators。
import os
from step1_config import *
from step4_split import X_train_s, X_test_s, y_train, y_test

os.chdir(os.path.dirname(os.path.abspath(__file__)))

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)
y_prob_rf = rf.predict_proba(X_test_s)[:, 1]

print("随机森林 测试集表现:")
print("  Accuracy :", round(accuracy_score(y_test, y_pred_rf), 4))
print("  Precision:", round(precision_score(y_test, y_pred_rf), 4))
print("  Recall   :", round(recall_score(y_test, y_pred_rf), 4))
print("  F1       :", round(f1_score(y_test, y_pred_rf), 4))
print("  AUC      :", round(roc_auc_score(y_test, y_prob_rf), 4))

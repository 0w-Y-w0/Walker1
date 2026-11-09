# step5_lr.py
# 用途: 训练逻辑回归作为可解释基线，输出标准准确率，观察线性边界在不平衡数据上的局限。
# 关键参数:
#   class_weight="balanced"  按类别频率自动加权，缓解不平衡
#   max_iter=1000            保证收敛（默认 100 可能不够并报警告）
#   C=1.0                    正则化强度倒数，越大拟合越强
# 预期输出: 打印 accuracy/precision/recall/f1/auc: 注意 precision 可能偏低（误报多）。
# 常见报错与解决:
#   - ConvergenceWarning -> 增大 max_iter 或减小 C（加强正则）;
#   - 概率输出全接近 0.5 -> 特征尺度不一，已用 StandardScaler 解决。
import os
from step1_config import *
from step4_split import X_train_s, X_test_s, y_train, y_test

os.chdir(os.path.dirname(os.path.abspath(__file__)))

lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                        random_state=RANDOM_STATE)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

print("逻辑回归 测试集表现:")
print("  Accuracy :", round(accuracy_score(y_test, y_pred_lr), 4))
print("  Precision:", round(precision_score(y_test, y_pred_lr), 4))
print("  Recall   :", round(recall_score(y_test, y_pred_lr), 4))
print("  F1       :", round(f1_score(y_test, y_pred_lr), 4))
print("  AUC      :", round(roc_auc_score(y_test, y_prob_lr), 4))
print(classification_report(y_test, y_pred_lr, target_names=["正常", "欺诈"]))

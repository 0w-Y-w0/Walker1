# step7_svm.py
# 用途: 用 RBF 核 SVM 学习非线性最大间隔边界，作为决策边界方法的代表。
# 关键参数:
#   C=1.0                 惩罚系数，越大越不容忍误分类（易过拟合）
#   gamma="scale"         核带宽，自动按特征尺度缩放（避免全连成一片）
#   probability=True      需要概率输出做 ROC/AUC（会增加训练时间）
#   class_weight="balanced"
# 预期输出: Recall 较高，但训练较慢; AUC 与随机森林接近。
# 常见报错与解决:
#   - 训练极慢（2 万样本）-> 可减少 n_samples 到 5000，或换 LinearSVC;
#   - probability fit 慢 -> 若只有决策函数可设 probability=False，用 decision_function
#     代替 predict_proba。
# 注意: SVM 在大数据上较慢，这里用子集加速演示（取前 8000 训练）
import os
from step1_config import *
from step4_split import X_train_s, X_test_s, y_train, y_test

os.chdir(os.path.dirname(os.path.abspath(__file__)))

X_tr_sub, _, y_tr_sub, _ = train_test_split(
    X_train_s, y_train, train_size=8000, random_state=RANDOM_STATE, stratify=y_train
)
svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
          probability=True, random_state=RANDOM_STATE)
svm.fit(X_tr_sub, y_tr_sub)
# 用同一放缩后的测试集评估
y_pred_svm = svm.predict(X_test_s)
y_prob_svm = svm.predict_proba(X_test_s)[:, 1]

print("SVM(RBF) 测试集表现:")
print("  Accuracy :", round(accuracy_score(y_test, y_pred_svm), 4))
print("  Precision:", round(precision_score(y_test, y_pred_svm), 4))
print("  Recall   :", round(recall_score(y_test, y_pred_svm), 4))
print("  F1       :", round(f1_score(y_test, y_pred_svm), 4))
print("  AUC      :", round(roc_auc_score(y_test, y_prob_svm), 4))

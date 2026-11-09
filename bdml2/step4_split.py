# step4_split.py
# 用途: 将数据划分为训练集(70%)、测试集(30%)，并做标准化（对 LR/SVM 尤其重要）。
# 关键参数:
#   test_size=0.3, stratify=y  分层采样，保证训练/测试的正负比例一致
#   scaler 用训练集 fit，transform 测试集（防信息泄露）
# 预期输出: 打印训练/测试集形状与正类占比，训练/测试占比均接近 1.97%。
# 常见报错与解决:
#   - DataConversionWarning -> 确保 X 为 float 数组，用 X.astype(np.float64);
#   - 测试集占比异常 -> 保证 stratify=y 而非 stratify=None（默认会破坏不平衡比例）。
import numpy as np
import os
from step1_config import *
from step2_make_data import df   # 数据集在 step2 中生成

os.chdir(os.path.dirname(os.path.abspath(__file__)))

X = df.drop(columns=["label"]).to_numpy().astype(np.float64)
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("训练集:", X_train_s.shape, "正类占比 {:.3%}".format(y_train.mean()))
print("测试集:", X_test_s.shape, "正类占比 {:.3%}".format(y_test.mean()))

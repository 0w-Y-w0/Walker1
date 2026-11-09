# step3_split_scale.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy()
y = df["y"].to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=482)          # 8:2 划分, 固定种子可复现

scaler = StandardScaler().fit(X_tr)                # 仅用训练集拟合, 避免信息泄露
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)
print("训练集:", X_tr_s.shape, "测试集:", X_te_s.shape)
print("标准化后特征均值≈0, 标准差≈1:", X_tr_s.mean(axis=0).round(2))

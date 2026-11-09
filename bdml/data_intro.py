# data_intro.py — 认识两种数据来源（先运行理解，再进入第四节实战）
from sklearn.datasets import fetch_california_housing
import numpy as np, pandas as pd

# 来源 1: 加州房价（联网下载, 约 2 万行）
cal = fetch_california_housing(as_frame=True)
print("加州房价 shape:", cal.data.shape, "目标列:", cal.target.name)
print(cal.data.head())

# 来源 2: 合成大数据（本地生成, 无需下载, 体现百万行量级）
rng = np.random.default_rng(482)
n = 1_000_000                    # 100 万行
X = rng.standard_normal((n, 5))
true_w = np.array([3.0, -2.0, 1.5, 0.5, -1.0])
y = X @ true_w + 1.0 + rng.normal(0, 0.5, n)      # 线性+噪声
print("合成数据 shape:", X.shape, "内存(约):", X.nbytes/1e6, "MB")
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df["y"] = y
print(df.describe().T[["mean", "std", "min", "max"]])

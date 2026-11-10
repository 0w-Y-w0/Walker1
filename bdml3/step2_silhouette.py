# step2_silhouette.py
# 用途: 在抽样数据上计算不同 K 的轮廓系数，取最大值对应的 K
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 生成与肘部实验相同的数据
X, _ = make_blobs(n_samples=20000, n_features=2, centers=5,
                  cluster_std=1.0, random_state=482)
Xs = StandardScaler().fit_transform(X)

# 对大数据抽样，避免轮廓系数计算过慢（O(n^2) 距离矩阵）
idx = np.random.RandomState(482).choice(len(Xs), 5000, replace=False)
X_sample = Xs[idx]

best_k, best_s = 2, -1
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=482, n_init=10).fit_predict(X_sample)
    s = silhouette_score(X_sample, labels)
    print(f"K={k}, silhouette={s:.3f}")
    if s > best_s:
        best_k, best_s = k, s
print("推荐 K =", best_k, "轮廓系数 =", round(best_s, 3))

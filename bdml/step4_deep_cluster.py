# step4_deep_cluster.py
# 用途: 在 AE 隐空间上跑 K-Means，并与"原始特征 K-Means"对比质量
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

H = np.load("latent_ae.npy")

# 重建"原始特征"侧的数据（与 step4_ae 完全一致的生成与标准化）
X, _ = make_blobs(n_samples=50000, n_features=5, centers=5,
                  cluster_std=1.0, random_state=482)
Xs = StandardScaler().fit_transform(X)

# 原始特征 K-Means（在标准化 Xs 上）
km_raw = KMeans(n_clusters=5, random_state=482, n_init=10).fit(Xs)
# 深度聚类: 隐空间上 K-Means
km_deep = KMeans(n_clusters=5, random_state=482, n_init=10).fit(H)

s_raw = silhouette_score(Xs[:5000], km_raw.labels_[:5000])
s_deep = silhouette_score(H[:5000], km_deep.labels_[:5000])
print(f"原始特征 K-Means 轮廓系数: {s_raw:.3f}")
print(f"AE 隐空间  K-Means 轮廓系数: {s_deep:.3f}")
print("提升幅度: {:.1%}".format((s_deep - s_raw) / abs(s_raw)))

# 保存标签可供可视化步骤使用
np.save("labels_raw.npy", km_raw.labels_)
np.save("labels_deep.npy", km_deep.labels_)
print("已保存 labels_raw.npy / labels_deep.npy")

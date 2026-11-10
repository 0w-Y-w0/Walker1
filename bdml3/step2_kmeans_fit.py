# step2_kmeans_fit.py
# 用途: 用选定 K=5 训练 K-Means，并在前 2 维散点上按簇着色
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 与前两节一致的数据
X, _ = make_blobs(n_samples=20000, n_features=2, centers=5,
                  cluster_std=1.0, random_state=482)
Xs = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=5, random_state=482, n_init=10)
labels = km.fit_predict(Xs)
centers = km.cluster_centers_

plt.figure(figsize=(6, 4))
sc = plt.scatter(Xs[:8000, 0], Xs[:8000, 1], c=labels[:8000],
                 cmap="tab10", s=4, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=120, label="质心")
plt.legend(); plt.title("K-Means (K=5)")
plt.tight_layout(); plt.savefig("fig_kmeans.png", dpi=120)
print("各簇样本数:", np.bincount(labels))
print("轮廓系数(抽样):",
      round(__import__("sklearn.metrics", fromlist=["silhouette_score"])
            .silhouette_score(Xs[:5000], labels[:5000]), 3))

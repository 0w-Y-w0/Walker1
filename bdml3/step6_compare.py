# step6_compare.py
# 用途: 把原始特征(降到2维)与 AE 隐空间的聚类结果并排画散点对比
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

H = np.load("latent_ae.npy")
labels_raw = np.load("labels_raw.npy")
labels_deep = np.load("labels_deep.npy")

# 重建原始特征（与 step4_ae 一致），再用 PCA 降到 2 维画散点
X, _ = make_blobs(n_samples=50000, n_features=5, centers=5,
                  cluster_std=1.0, random_state=482)
Xs = StandardScaler().fit_transform(X)
X2 = PCA(n_components=2).fit_transform(Xs)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X2[:8000, 0], X2[:8000, 1], c=labels_raw[:8000],
                cmap="tab10", s=3, alpha=0.6)
axes[0].set_title("原始特征  K-Means")
axes[1].scatter(H[:8000, 0], H[:8000, 1], c=labels_deep[:8000],
                cmap="tab10", s=3, alpha=0.6)
axes[1].set_title("AE 隐空间  K-Means")
plt.tight_layout(); plt.savefig("fig_compare.png", dpi=120)
print("已保存对比图 fig_compare.png")

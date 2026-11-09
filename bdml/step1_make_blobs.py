# step1_make_blobs.py
# 用途: 生成 100,000 条、5 个真实簇、含少量噪声的大数据，用于验证聚类算法在"大样本"上的表现
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文正常显示
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.RandomState(482)
X, y_true = make_blobs(
    n_samples=100_000,        # 样本量：十万级，体现"大数据"
    n_features=5,             # 特征维数（先高维，后续降维可视化）
    centers=5,                # 真实簇数（仅供评估参考，聚类时不告诉模型）
    cluster_std=1.0,          # 簇内标准差，越大越重叠
    random_state=rng,
)
print("X shape:", X.shape, "标签分布:", np.bincount(y_true))

# 特征标准化：聚类前的关键一步
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print("标准化后 各维均值=", np.round(Xs.mean(axis=0), 3),
      " 标准差=", np.round(Xs.std(axis=0), 3))

# 抽前 5000 点画二维散点（取前 2 维预览簇结构）
plt.figure(figsize=(5, 5))
plt.scatter(Xs[:5000, 0], Xs[:5000, 1], s=1, alpha=0.3)
plt.title("合成聚类数据预览(前2维, 抽样5000点)")
plt.tight_layout(); plt.savefig("blobs_preview.png", dpi=120)
print("已保存预览图 blobs_preview.png")

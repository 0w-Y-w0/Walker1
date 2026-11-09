# step3_dbscan.py
# 用途: 在含噪声、非球形数据上演示 DBSCAN，对比 K-Means 的局限
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 月牙形（非凸）数据: K-Means 会失败, DBSCAN 能发现
Xm, _ = make_moons(n_samples=2000, noise=0.05, random_state=482)
Xm = StandardScaler().fit_transform(Xm)

db = DBSCAN(eps=0.2, min_samples=10)     # 核心参数
labels = db.fit_predict(Xm)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print("簇数:", n_clusters, "噪声点:", n_noise)

plt.figure(figsize=(6, 4))
# 噪声点(-1)用灰色, 其余按簇着色
unique = set(labels)
for lab in unique:
    mask = labels == lab
    color = "gray" if lab == -1 else None
    plt.scatter(Xm[mask, 0], Xm[mask, 1], s=4,
                c=color, cmap="tab10", label=("噪声" if lab == -1 else f"簇{lab}"))
plt.legend(); plt.title("DBSCAN on moons")
plt.tight_layout(); plt.savefig("fig_dbscan.png", dpi=120)
print("已保存 fig_dbscan.png")

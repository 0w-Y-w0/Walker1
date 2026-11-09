# step2_elbow.py
# 用途: 在 2..10 范围内计算各 K 的 inertia，画肘部图确定合适簇数
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X, _ = make_blobs(n_samples=20000, n_features=2, centers=5,
                  cluster_std=1.0, random_state=482)
Xs = StandardScaler().fit_transform(X)

Ks = range(2, 11)
inertias = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=482, n_init=10)
    km.fit(Xs)
    inertias.append(km.inertia_)     # WCSS(簇内平方和)

plt.figure(figsize=(6, 4))
plt.plot(list(Ks), inertias, "o-")
plt.xlabel("K (簇数)"); plt.ylabel("Inertia (WCSS)")
plt.title("肘部法则 Elbow Method")
plt.grid(True)
plt.tight_layout(); plt.savefig("fig_elbow.png", dpi=120)
print("inertias:", [round(i, 1) for i in inertias])

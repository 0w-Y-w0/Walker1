# step4_real_e2e.py
# 用途: 对真实批发客户数据做完整聚类流水线: 标准化 -> 选K -> 聚类 -> 画像 -> 决策
# 注: OpenML 502 已失效, 数据来自 UCI 直链(4.1.2 已下载 wholesale.csv); 若无则降级 make_blobs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# —— 数据: 优先真实, 降级合成 ——
try:
    df_raw = pd.read_csv("wholesale.csv")
    num_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    X_raw = df_raw[num_cols].values
    src = "Wholesale(真实)"
except Exception as e:
    print("真实数据不可用:", e)
    X_raw, _ = make_blobs(n_samples=2000, n_features=5, centers=4,
                          cluster_std=1.2, random_state=482)
    num_cols = [f"特征{i}" for i in range(X_raw.shape[1])]
    src = "make_blobs(合成)"

Xs = StandardScaler().fit_transform(X_raw)

# —— 选 K: 轮廓系数最高值 ——
best_k, best_s = 2, -1
for k in range(2, 8):
    sample = Xs[:5000] if len(Xs) > 5000 else Xs
    s = silhouette_score(sample,
                         KMeans(n_clusters=k, random_state=482,
                                n_init=10).fit_predict(sample))
    if s > best_s:
        best_k, best_s = k, s

km = KMeans(n_clusters=best_k, random_state=482, n_init=10).fit(Xs)
df = pd.DataFrame(X_raw, columns=num_cols)          # 用原始量纲看画像（非标准化）
df["cluster"] = km.labels_
profile = df.groupby("cluster").mean().round(1)
print(f"数据源={src}  选K={best_k}  轮廓={best_s:.3f}")
print("各类支出画像(原始量纲):\n", profile)

# —— 画像 -> 决策: 以"各类总支出均值"高低粗略分层 ——
overall = df[num_cols].mean()
for c, row in profile.iterrows():
    above = [col for col in num_cols if row[col] > overall[col]]
    level = "高价值" if row[num_cols].mean() > overall.mean() else "一般/低价值"
    print(f"簇{c}[{level}] 高于均值的品类: {above if above else '无'} -> "
          f"{'重点维系/差异化供货' if level == '高价值' else '标准化服务/培育'}")

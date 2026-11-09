# step1_generate.py — 向量化生成百万行回归数据（避免 Python 循环）
import numpy as np, pandas as pd, time

rng = np.random.default_rng(0)
n, d = 1_000_000, 8

t0 = time.time()
# 向量化：一次性生成全部特征与目标，无任何 Python for 循环
X = rng.standard_normal((n, d))                      # (100万, 8) 标准正态特征
true_w = rng.uniform(-2, 2, size=d)                  # 真实权重（用于后面验证模型能否学回）
y = X @ true_w + 5.0                                 # 线性部分
# 加非线性项与噪声，让数据更"真实"、更具挑战
y = y + 0.3 * np.sin(X[:, 0] * 3.0) + rng.normal(0, 1.0, n)
print(f"生成 {n} 行耗时: {time.time()-t0:.2f}s")
print(f"特征矩阵内存: {X.nbytes/1e6:.1f} MB, 标签: {y.nbytes/1e6:.1f} MB")

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
df["y"] = y
df.to_parquet("bigdata_reg.parquet")                 # 列式格式，后续读取更快更小
print("已保存 bigdata_reg.parquet，大小约:",
      round(__import__("os").path.getsize("bigdata_reg.parquet")/1e6, 1), "MB")
print("真实权重 true_w:", np.round(true_w, 3))

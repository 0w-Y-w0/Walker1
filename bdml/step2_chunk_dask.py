# step2_chunk_dask.py
import pandas as pd, numpy as np

# 方法 A: pandas 分块—逐块处理, 内存常驻很小
chunk_sum, chunk_n = 0.0, 0
for chunk in pd.read_csv("bigdata_reg.csv", chunksize=200_000):
    chunk_sum += chunk["y"].sum()
    chunk_n += len(chunk)
print("pandas 分块 y 均值:", chunk_sum / chunk_n)

# 方法 B: Dask—把计算图推迟到真正需要时执行（需安装 dask）
import dask.dataframe as dd
ddf = dd.read_parquet("bigdata_reg.parquet")
print("Dask 惰性均值:", float(ddf["y"].mean().compute()))    # .compute() 才真正触发计算

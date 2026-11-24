# data_split.py
# 步骤1: 数据预处理与归一化
# 用途: 对原始序列做 z-score 归一化（仅用训练集统计量），按时间切分
#       train / val / test，并预留 seq_len/pred_len 长度供滑窗取输入。
# 关键参数:
#   seq_len=96    回溯窗口, 对应 ETTh1 的 4 天(24h*4)
#   pred_len=24   预测未来 24 步 = 1 天
#   切分比例 7:1:2
# 预期输出: 各集合长度; 归一化后训练集均值≈0、标准差≈1。
# 常见报错与解决:
#   - ValueError 输入序列太短 -> ETTh1 无此问题; seq_len 过大时调小;
#   - 归一化出现 nan -> 某通道标准差为 0(常量列), 已在 std 后加 1e-8。
# 重要概念: 窗口长度两个超参数——seq_len 应至少覆盖 1~2 个主要周期
#   (日周期24/周周期168); pred_len 由业务决定(负荷调度通常预测 24h)。
import numpy as np
import pandas as pd

def time_split(data, seq_len, pred_len, train_ratio=0.7, val_ratio=0.1):
    # 仅用训练段计算均值/标准差，防止未来信息泄漏
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    mu = data[:n_train].mean(axis=0)
    sigma = data[:n_train].std(axis=0) + 1e-8
    norm = (data - mu) / sigma
    train = norm[:n_train + seq_len]           # 多留 seq_len 供滑窗取输入
    val = norm[n_train:n_train + n_val + seq_len + pred_len]
    test = norm[n_train + n_val:]
    return train, val, test, mu, sigma

df = pd.read_csv("ETTh1.csv")
data = df.drop(columns=["date"]).values.astype("float32")

seq_len, pred_len = 96, 24
train, val, test, mu, sigma = time_split(data, seq_len, pred_len)
print("train/val/test 长度:", len(train), len(val), len(test))
print("训练集均值(应≈0):", np.round(mu, 3))
print("训练集标准差(应≈1):", np.round(sigma, 3))

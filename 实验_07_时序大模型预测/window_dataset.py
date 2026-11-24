# window_dataset.py
# 步骤2: 构造滑动窗口数据集
# 用途: 把连续序列切成监督学习样本 (x[L,C] -> y[H,C])。
# 关键参数: SlidingWindowDataset(data, seq_len, pred_len);
#           train 用 batch_size=64 + shuffle=True, val 不打乱。
# 预期输出: len(dataset); 单样本 x.shape=(96,7), y.shape=(24,7) (ETTh1 7通道)。
# 常见报错与解决:
#   - Windows 上 DataLoader 多进程报 BrokenPipeError -> num_workers=0;
#   - CUDA out of memory -> batch_size 调 32/16 或缩短 seq_len。
# 关于 shuffle 的重要讨论:
#   train 打乱以稳定优化, 但 val/test 必须 shuffle=False 保持时间顺序——
#   否则发生"用未来样本指导过去评估"的隐性泄漏; 时序 CV 应该用
#   滚动窗口交叉验证(rolling-origin)而非随机 k 折。
import torch
from torch.utils.data import Dataset, DataLoader
from data_split import time_split
import pandas as pd

class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.x, self.y = [], []
        T = data.shape[0]
        for i in range(T - seq_len - pred_len + 1):
            self.x.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len:i+seq_len+pred_len])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

df = pd.read_csv("ETTh1.csv")
data = df.drop(columns=["date"]).values.astype("float32")
seq_len, pred_len = 96, 24
train, val, test, mu, sigma = time_split(data, seq_len, pred_len)

train_ds = SlidingWindowDataset(train, seq_len, pred_len)
val_ds   = SlidingWindowDataset(val,   seq_len, pred_len)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)
print("训练样本数:", len(train_ds))
x0, y0 = train_ds[0]
print("单样本 x/y 形状:", tuple(x0.shape), tuple(y0.shape))

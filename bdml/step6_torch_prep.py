# step6_torch_prep.py
import numpy as np, pandas as pd, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy(); y = df["y"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=482)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

# 转为 PyTorch 张量；y 增加一维以匹配网络输出形状 (n,1)
Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_te_s, dtype=torch.float32)
yte = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(Xtr, ytr)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
print("每 epoch 批次数:", len(train_loader), " 特征维度:", Xtr.shape[1])

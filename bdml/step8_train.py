# step8_train.py — 合并 step6/step7 代码 + 训练循环（MSE 损失 + Adam + 早停）
import numpy as np, pandas as pd, time, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- 数据准备（来自 step6） ----------
df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy(); y = df["y"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_te_s, dtype=torch.float32)
yte = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(Xtr, ytr)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)

# ---------- 网络定义（来自 step7） ----------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim=8, hidden=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

model = MLPRegressor(in_dim=8, hidden=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # 自适应优化器

# ---------- 训练循环 + 早停 ----------
best_val, patience, wait = float("inf"), 5, 0
t0 = time.time()
for epoch in range(1, 31):
    model.train(); running = 0.0
    for Xb, yb in train_loader:                # 小批量训练
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * Xb.size(0)
    train_mse = running / len(train_loader.dataset)

    # 每轮在测试集评估（作业验证集做早停）
    model.eval()
    with torch.no_grad():
        val_mse = criterion(model(Xte), yte).item()
    print(f"Epoch {epoch:2d} | train MSE {train_mse:.3f} | val MSE {val_mse:.3f}")

    if val_mse < best_val - 1e-4:              # 验证集改善
        best_val, patience, wait = best_val, patience, 0
        best_val = val_mse
        torch.save(model.state_dict(), "mlp_best.pt")  # 保存最优
    else:
        wait += 1
        if wait >= patience:                   # 连续 5 轮无改善 -> 早停
            print("早停! 验证损失连续", patience, "轮未改善")
            break

print(f"总耗时: {time.time()-t0:.1f}s")
model.load_state_dict(torch.load("mlp_best.pt"))
print("已加载最优模型，最佳 val MSE =", round(best_val, 3))

# step4_ae.py
# 用途: 用 PyTorch 训练 AE，把高维特征压缩为低维表示（去噪、保留结构）
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1) 数据
X, _ = make_blobs(n_samples=50000, n_features=5, centers=5,
                  cluster_std=1.0, random_state=482)
Xs = StandardScaler().fit_transform(X)
X_t = torch.tensor(Xs, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_t), batch_size=512, shuffle=True)

# 2) 模型: 编码器 5->16->2, 解码器 2->16->5
class AE(nn.Module):
    def __init__(self, d_in=5, d_hidden=16, d_lat=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(),
                                 nn.Linear(d_hidden, d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, d_hidden), nn.ReLU(),
                                 nn.Linear(d_hidden, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))

model = AE()
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 3) 训练循环（CPU 即可，约 1~2 分钟）
device = "cpu"
model.to(device)
for epoch in range(1, 31):
    tot = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        opt.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward(); opt.step()
        tot += loss.item() * len(batch)
    if epoch % 5 == 0:
        print(f"epoch {epoch:02d}  loss={tot/len(X_t):.4f}")

# 4) 提取隐表示（丢弃解码器）
model.eval()
with torch.no_grad():
    H = model.enc(X_t).numpy()      # (n, 2) 低维隐空间
np.save("latent_ae.npy", H)
print("隐空间形状:", H.shape)

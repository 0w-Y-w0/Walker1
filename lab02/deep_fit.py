import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 解决 matplotlib 中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("synthetic.csv")
X_np = df["x"].values.astype("float32").reshape(-1, 1)
y_np = df["y"].values.astype("float32").reshape(-1, 1)

# 转为 PyTorch 张量
X_t = torch.from_numpy(X_np)
y_t = torch.from_numpy(y_np)


# 定义两层全连接网络: 输入 1 -> 隐藏层 32(ReLU) -> 输出 1
class TwoLayerNet(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model = TwoLayerNet()
criterion = nn.MSELoss()                                # 均方差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    y_pred = model(X_t)
    loss = criterion(y_pred, y_t)
    optimizer.zero_grad()          # 清空历史梯度
    loss.backward()                # 反向传播
    optimizer.step()               # 更新参数
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}]  loss = {loss.item():.4f}")

# 在网格上预测
x_grid = np.linspace(-3, 3, 200, dtype="float32").reshape(-1, 1)
with torch.no_grad():
    y_deep = model(torch.from_numpy(x_grid)).numpy().ravel()

# 计算训练 MSE
with torch.no_grad():
    train_mse = criterion(model(X_t), y_t).item()
print(f"深度网络训练 MSE = {train_mse:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(X_np, y_np, s=18, alpha=0.6, label="观测数据(含噪声)")
plt.plot(x_grid, y_deep, color="green", linewidth=2, label="神经网络拟合")
plt.title("深度模型: 两层全连接网络")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("deep_fit.png", dpi=120); plt.show()
print("已保存 deep_fit.png")

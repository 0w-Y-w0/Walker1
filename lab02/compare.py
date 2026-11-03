import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import os

# 无论从哪个目录运行，都切换到本脚本所在目录，保证相对路径可用
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 解决 matplotlib 中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("synthetic.csv")
X = df[["x"]].values.astype("float32")
y = df["y"].values.astype("float32").reshape(-1, 1)

# 浅层：线性回归
lr = LinearRegression().fit(X, y.ravel())


# 深度（复用 B3 的网络定义，这里简化重训一次）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


net = Net()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
crit = nn.MSELoss()
Xt = torch.from_numpy(X)
yt = torch.from_numpy(y)
for _ in range(1000):
    opt.zero_grad()
    loss = crit(net(Xt), yt)
    loss.backward()
    opt.step()

x_grid = np.linspace(-3, 3, 200, dtype="float32").reshape(-1, 1)
with torch.no_grad():
    y_deep = net(torch.from_numpy(x_grid)).numpy().ravel()

plt.figure(figsize=(8, 5))
plt.scatter(X, y, s=18, alpha=0.6, label="观测数据")
plt.plot(x_grid.ravel(), lr.predict(x_grid), color="red", lw=2, label="线性回归(浅层)")
plt.plot(x_grid.ravel(), y_deep, color="green", lw=2, label="神经网络(深度)")
plt.title("浅层 vs 深度模型拟合对比")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("compare.png", dpi=120)
plt.show()
print("对比图已保存 compare.png")

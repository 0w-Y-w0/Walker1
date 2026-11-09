# step9_eval_plot.py — 测试集评估 + 预测 vs 真实散点图（合并 step6/step7 数据与模型定义）
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")                        # 无界面环境也能存图
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文正常显示
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据准备（与训练时完全一致的划分与标准化） ----------
df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy(); y = df["y"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_te_s = scaler.transform(X_te)

Xte = torch.tensor(X_te_s, dtype=torch.float32)
yte = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

# ---------- 模型定义与加载最优权重 ----------
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
model.load_state_dict(torch.load("mlp_best.pt"))
model.eval()

with torch.no_grad():
    y_pred = model(Xte).numpy().ravel()

mse = mean_squared_error(yte.numpy(), y_pred)
mae = mean_absolute_error(yte.numpy(), y_pred)
r2 = r2_score(yte.numpy(), y_pred)
print(f"MLP 测试结果 -> MSE={mse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

# 抽 2000 个点画预测 vs 真实散点
idx = np.random.default_rng(6).choice(len(y_pred), 2000, replace=False)
plt.figure(figsize=(5, 5))
plt.scatter(yte.numpy()[idx], y_pred[idx], s=6, alpha=0.4)
lim = [min(yte.numpy().min(), y_pred.min()), max(yte.numpy().max(), y_pred.max())]
plt.plot(lim, lim, "r--", lw=1)               # 对角线 = 完美预测参考线
plt.xlabel("真实值 y")
plt.ylabel("预测值 y_hat")
plt.title(f"MLP 预测 vs 真实 (R2={r2:.3f})")
plt.tight_layout(); plt.savefig("mlp_pred_vs_true.png", dpi=120)
print("已保存散点图 mlp_pred_vs_true.png")

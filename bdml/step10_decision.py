# step10_decision.py — 预测值 -> 业务决策映射（房价预测 -> 买卖决策）
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- 数据与模型（合并 step6/step7 代码，避免跨文件导入问题） ----------
df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy(); y = df["y"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
Xte = torch.tensor(scaler.transform(X_te), dtype=torch.float32)

class MLPRegressor(nn.Module):
    def __init__(self, in_dim=8, hidden=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

model = MLPRegressor(in_dim=8, hidden=64)
model.load_state_dict(torch.load("mlp_best.pt")); model.eval()
with torch.no_grad():
    p_hat = model(Xte).numpy().ravel()

# 模拟一批房源的"市场挂牌价"（这里用测试集真实价加随机扰动代替）
rng = np.random.default_rng(7)
p_list = (Xte[:, 0].numpy()) * 0.0 + rng.uniform(1.0, 5.0, size=len(Xte))

# 规则: 预测价相对挂牌价低估 >15% 买入, 高估 >15% 卖出/不买, 否则观望
gap = (p_hat - p_list) / p_hat
decision = np.where(gap > 0.15, "买入",
           np.where(gap < -0.15, "卖出/不买", "观望"))
tbl = pd.DataFrame({"挂牌价": p_list.round(2), "预测价": p_hat.round(2),
                    "偏差": gap.round(3), "决策": decision})
print(tbl.head(10))
print("\n决策分布:\n", tbl["决策"].value_counts())

# step7_mlp_model.py
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim=8, hidden=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),        # 输入层 -> 隐藏层
            nn.ReLU(),                        # 非线性激活
            nn.Dropout(drop),                 # 随机置零，抑制过拟合
            nn.Linear(hidden, hidden // 2),   # 隐藏层 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),        # 隐藏层 -> 输出(单值)
        )

    def forward(self, x):
        return self.net(x)

model = MLPRegressor(in_dim=8, hidden=64)
print(model)
print("可训练参数总量:", sum(p.numel() for p in model.parameters()))

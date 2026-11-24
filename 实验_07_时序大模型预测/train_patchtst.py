# train_patchtst.py
# 步骤5: 训练
# 用途: 用 MSE 损失在 train 上训练, 在 val 上选最优, 保存权重。
# 关键参数:
#   lr=1e-3, epochs=20, 优化器 AdamW, 损失 MSELoss
#   已有 NVIDIA GPU 约几分钟; CPU 可把 epochs 调到 5~10
# 预期输出: 每轮打印 train loss 与 val loss(应同步下降),
#           最后保存 checkpoints/patchtst_best.pt。
# 常见报错与解决:
#   - 训练 loss 不降: 学习率过大/过小; 检查是否归一化、label 是否归一化
#     (本流程 x/y 统一用归一化数据, 无需额外处理);
#   - Windows DataLoader 多进程卡死: num_workers=0(已在 window_dataset 设置);
#   - 训练过慢(CPU): epochs 降到 5、d_model 降到 64、n_layers 降到 2。
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from window_dataset import train_dl, val_dl, train_ds, val_ds
from patch_tst import PatchTST

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatchTST(n_vars=2, patch_len=16, stride=8, d_model=128,
                 n_heads=8, n_layers=3, dropout=0.1, pred_len=24).to(device)

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)
best_val = float("inf")
history = {"train": [], "val": []}          # 步骤8: 记录每轮损失供可视化
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, 21):
    model.train()
    train_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_ds)
    history["train"].append(train_loss)
    history["val"].append(val_loss)
    print(f"Epoch {epoch:02d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "checkpoints/patchtst_best.pt")
        print("  -> 保存最优模型")

# 持久化训练历史(步骤8 可视化用)
import json  # noqa: E402
with open("checkpoints/history.json", "w", encoding="utf-8") as f:
    json.dump(history, f)

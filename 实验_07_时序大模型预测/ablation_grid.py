# ablation_grid.py
# 步骤7: 超参数调优与消融实验（进阶）
# 用途: 通过系统性的参数与结构消融, 理解哪些设计真正影响预测精度——
#       培养"做实验而非跑脚本"的研究习惯。
# 关键参数: patch_len ∈ {8,16,32}, d_model ∈ {64,128}, n_layers ∈ {2,3,6},
#           stride(决定 patch 数量)。
# 预期输出: 一张"不同配置下的 val MSE"对比表 + 指出最敏感超参;
#           附 2 组结构消融(去位置编码 / 池化改 last patch)。
# 常见报错与解决:
#   - 网格搜索耗时过长(CPU): 只保留前 2~3 组或 epochs=5(本机 GPU 全跑);
#   - n_heads=8 与 d_model=64 不整除: 改 n_heads=4 或保持 d_model 为 8 的倍数。
# 消融建议(教程): (1) 去位置编码 (2) mean 池化改取最后一个 patch
#   —— 两项均已实现, 结果可写进实验报告"讨论"部分。
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from window_dataset import train_dl, val_dl, train_ds, val_ds
from patch_tst import PatchTST

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.MSELoss()

def train_model(model, train_dl, val_dl, epochs=10):
    optimizer = AdamW(model.parameters(), lr=1e-3)
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        best_val = min(best_val, val_loss)
    return best_val

# 网格: 4 组超参 + 2 组结构消融
configs = [
    dict(patch_len=16, stride=8, d_model=128, n_layers=3),              # 基准
    dict(patch_len=8,  stride=4, d_model=128, n_layers=3),              # 更碎 patch
    dict(patch_len=16, stride=8, d_model=64,  n_layers=2),              # 小模型
    dict(patch_len=32, stride=16, d_model=128, n_layers=6),             # 大 patch 深层
    dict(patch_len=16, stride=8, d_model=128, n_layers=3,
         use_pos_enc=False),                                            # 消融: 去位置编码
    dict(patch_len=16, stride=8, d_model=128, n_layers=3, pool="last"), # 消融: last 池化
]

print(f"{'配置':<44} val MSE")
print("-" * 60)
results = []
for i, cfg in enumerate(configs, 1):
    m = PatchTST(n_vars=2, pred_len=24, n_heads=8, dropout=0.1, **cfg).to(device)
    best = train_model(m, train_dl, val_dl, epochs=10)
    results.append((cfg, best))
    print(f"#{i} {cfg} -> {best:.4f}")

best_cfg, best_mse = min(results, key=lambda r: r[1])
print("\n=== 结论 ===")
print(f"最优配置: {best_cfg} (val MSE {best_mse:.4f})")
diff = {("use_pos_enc", "pool"): None}
for (cfg, mse), (base_cfg, base_mse) in [(results[4], results[0]), (results[5], results[0])]:
    tag = "去位置编码" if cfg.get("use_pos_enc") is False else "池化改 last"
    print(f"消融[{tag}]: val MSE {base_mse:.4f} -> {mse:.4f} "
          f"({'变差' if mse > base_mse else '持平/变好'} {abs(mse-base_mse)/base_mse:+.1%})")

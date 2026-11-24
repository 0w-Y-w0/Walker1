# plot_pred_sample.py
# 5.1 预测结果可视化（历史 + 未来）
# 用途: 把某条测试样本的预测曲线与真实曲线画在一起, 直观判断模型
#       是否学到了趋势与周期。
# 关键参数: 取 preds[0]、gts[0] 的通道 0; 横轴为预测 horizon 步。
# 预期输出: figures/pred_sample.png——"真值(蓝) vs 预测(橙)"折线,
#           曲线应大体贴合。
# 常见报错与解决:
#   - 中文方框: plt.rcParams 设置 SimHei(已加);
#   - 曲线完全对不上: 检查反归一化是否用了同一 mu/sigma;
#   - 想看其它样本/通道: 改索引 preds[i, :, j] 即可。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import torch
from torch.utils.data import DataLoader as DL
from window_dataset import SlidingWindowDataset
from data_split import time_split
from patch_tst import PatchTST
import pandas as pd

seq_len, pred_len = 96, 24
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("ETTh1.csv")
data = df.drop(columns=["date"]).values.astype("float32")
train, val, test, mu, sigma = time_split(data, seq_len, pred_len)

model = PatchTST(n_vars=2, patch_len=16, stride=8, d_model=128,
                 n_heads=8, n_layers=3, dropout=0.1, pred_len=pred_len).to(device)
model.load_state_dict(torch.load("checkpoints/patchtst_best.pt",
                                 map_location=device))

test_ds = SlidingWindowDataset(test, seq_len, pred_len)
test_dl = DL(test_ds, batch_size=64, shuffle=False)
model.eval()
preds, gts = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds.append(model(xb.to(device)).cpu())
        gts.append(yb.cpu())
preds = torch.cat(preds).numpy() * sigma + mu   # 反归一化
gts = torch.cat(gts).numpy() * sigma + mu

plt.figure(figsize=(10, 4))
plt.plot(gts[0, :, 0], label="真实值（Ground Truth）", color="steelblue", linewidth=2)
plt.plot(preds[0, :, 0], label="预测值（PatchTST）", color="darkorange",
         linewidth=2, linestyle="--")
plt.title("测试样本预测: 未来 24 步（通道 0）")
plt.xlabel("预测步长 horizon"); plt.ylabel("数值（反归一化）")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/pred_sample.png", dpi=120)
print("已保存 figures/pred_sample.png")

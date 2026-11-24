# evaluate_eval.py
# 步骤6: 预测、反归一化与评估
# 用途: 在 test 段做多步预测, 反归一化回原始量纲, 计算 MAE/MSE/RMSE,
#       并与 ARIMA baseline 对比。
# 关键参数: mu、sigma（来自 time_split 的训练段统计量）。
# 预期输出: 打印深度模型与 ARIMA 在测试首段的 MAE / RMSE; 曲线图见第五节。
# 常见报错与解决:
#   - 反归一化后数值异常大: 确认 sigma 未含 1e-8 之外的额外缩放;
#     预测与真值必须使用同一 mu/sigma;
#   - ARIMA 与深度模型对比口径不同（单通道 vs 多通道）: 本对比仅作趋势性
#     参考, 论文级对比需统一 horizon 与通道, 报告中应说明差异。
import numpy as np
import torch
from torch.utils.data import DataLoader as DL
from window_dataset import SlidingWindowDataset
from data_split import time_split
from arima_baseline import arima_baseline
from patch_tst import PatchTST
import pandas as pd

seq_len, pred_len = 96, 24
device = "cuda" if torch.cuda.is_available() else "cpu"

# 重新取得 test 段与归一化统计量(与训练时完全同一套切分)
df = pd.read_csv("ETTh1.csv")
data = df.drop(columns=["date"]).values.astype("float32")
train, val, test, mu, sigma = time_split(data, seq_len, pred_len)

model = PatchTST(n_vars=2, patch_len=16, stride=8, d_model=128,
                 n_heads=8, n_layers=3, dropout=0.1, pred_len=pred_len).to(device)
model.load_state_dict(torch.load("checkpoints/patchtst_best.pt",
                                 map_location=device))

def evaluate(model, test_data, seq_len, pred_len, mu, sigma, batch_size=64):
    test_ds = SlidingWindowDataset(test_data, seq_len, pred_len)
    test_dl = DL(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds.append(model(xb).cpu())
            gts.append(yb.cpu())   # 教程笔误: yb 在 GPU 上, 需先 .cpu() 才能转 numpy
    preds = torch.cat(preds).numpy()          # [N, H, V]
    gts = torch.cat(gts).numpy()
    # 反归一化（对每个通道各自用 mu/sigma）
    preds = preds * sigma + mu
    gts = gts * sigma + mu
    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))
    return preds, gts, mae, rmse

preds, gts, mae, rmse = evaluate(model, test, seq_len, pred_len, mu, sigma)
print(f"深度模型(测试段) MAE={mae:.3f} RMSE={rmse:.3f}")

# 与 ARIMA 对比（仅通道0, 取测试段前 pred_len 步作对照）
test_ch0 = data[int(len(data) * 0.8):, 0]
arima_pred = arima_baseline(data[int(len(data) * 0.7):int(len(data) * 0.8), 0], pred_len)
gt_arima = test_ch0[:pred_len]
mae_arima = np.mean(np.abs(arima_pred - gt_arima))
rmse_arima = np.sqrt(np.mean((arima_pred - gt_arima) ** 2))
print(f"ARIMA(通道0)     MAE={mae_arima:.3f} RMSE={rmse_arima:.3f}")

# plot_loss_curve.py
# 步骤8: 训练过程监控（学习率、梯度、早停）
# 用途: 读取训练时收集的 history.json, 绘制 train/val 损失曲线,
#       确认模型"健康收敛"而非盲目等它跑完。
# 关键参数: history.json 的 train/val 数组。
# 预期输出: figures/loss_curve.png——train 与 val 同步下降且 val 出现拐点
#           (本实验 val 最优出现在第 11 轮附近, 之后回升即早停点)。
# 常见报错与解决:
#   - val loss 后期回升而 train 仍降 = 过拟合: 增大 dropout、减小模型、
#     或增加训练数据(合成数据可把 n 调到 5000+);
#   - loss 变 nan = 梯度爆炸: 在 loss.backward() 后、optimizer.step() 前
#     加 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0);
#   - 无显示环境: 本脚本用 Agg 后端直接存文件, plt.show() 仅占位。
import json
import matplotlib
matplotlib.use("Agg")          # 无 GUI 环境直接出图
import matplotlib.pyplot as plt

with open("checkpoints/history.json", encoding="utf-8") as f:
    history = json.load(f)

plt.figure(figsize=(8, 4))
plt.plot(history["train"], label="train MSE", color="steelblue")
plt.plot(history["val"], label="val MSE", color="darkorange")
plt.axvline(int(history["val"].index(min(history["val"]))), color="red",
            ls="--", alpha=0.6, label="早停点(val 最优)")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True, alpha=0.3)
plt.title("训练/验证损失曲线")
plt.tight_layout()
plt.savefig("figures/loss_curve.png", dpi=120)
print("已保存 figures/loss_curve.png; val 最优 epoch =",
      history["val"].index(min(history["val"])) + 1)

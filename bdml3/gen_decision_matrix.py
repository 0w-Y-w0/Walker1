# gen_decision_matrix.py
# 用途: 复现教程图 03-3 "客户分群决策矩阵"四象限示意图, 保存为 PNG。
#       色块/文字/位置与教程一致: 左上流失风险挽留(紫)、右上低价值低成本服务(橙)、
#       左下高价值重点维护(绿)、右下潜力培育(蓝)。
# 输出: fig_decision_matrix.png (150 dpi)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)

# 四象限色块: (x0, y0, 宽, 高, 颜色, 文字)
quads = [
    (0, 1, 1, 1, "#E6DFF4", "流失风险\n挽留"),        # 左上 紫
    (1, 1, 1, 1, "#FBE3C6", "低价值\n低成本服务"),    # 右上 橙
    (0, 0, 1, 1, "#D5EFD2", "高价值\n重点维护"),      # 左下 绿
    (1, 0, 1, 1, "#D3E2F4", "潜力\n培育"),            # 右下 蓝
]
for x0, y0, w, h, color, text in quads:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=color,
                               edgecolor="white", linewidth=2))
    ax.text(x0 + 0.5, y0 + 0.5, text, ha="center", va="center",
            fontsize=15, color="#333333", linespacing=1.8)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("消费金额（低 → 高）", fontsize=11, color="#555")
ax.set_ylabel("消费频率（低 → 高）", fontsize=11, color="#555")
ax.set_title("客户分群决策矩阵", fontsize=16, pad=14, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_decision_matrix.png", dpi=150)
print("已保存 fig_decision_matrix.png")

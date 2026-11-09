# step3_eda.py
# 用途: 通过可视化直观感受"少数类被多数类淹没"，并观察特征区分度。
# 关键参数: 取两个最具信息量的特征 f0、f1 做散点（仅为示意，真实区分需多维）。
# 预期输出: 一张散点图，蓝色为正常、红色为欺诈，可见红点极稀疏且散布。
# 常见报错与解决:
#   - 中文标题乱码 -> 检查 step1_config 字体设置;
#   - 图不显示 -> Jupyter 中用 %matplotlib inline（脚本则 plt.show()）。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from step1_config import *   # RANDOM_STATE、绘图样式、pandas 等全部就绪
from step2_make_data import df   # 数据集在 step2 中生成（导入时自动执行）

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.figure(figsize=(7, 5))
normal = df[df.label == 0]
fraud = df[df.label == 1]

plt.scatter(normal["f0"], normal["f1"], s=8, alpha=0.25, label="正常(0)",
            color="#41776a")
plt.scatter(fraud["f0"], fraud["f1"], s=18, alpha=0.85, label="欺诈(1)",
            color="#d66276")
plt.xlabel("f0"); plt.ylabel("f1")
plt.title("合成信用卡交易: 正常 vs 欺诈 (极度不平衡)")
plt.legend()
plt.tight_layout(); plt.savefig("fig_imbalance.png", dpi=120)
print("正常/欺诈样本数:", len(normal), "/", len(fraud), "-> 已保存 fig_imbalance.png")

# step5_decision.py
# 用途: 给每个簇计算画像，并据此生成差异化运营建议（决策矩阵）
import numpy as np
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 假设两维含义: dim0=消费频率维, dim1=消费金额维（隐空间可解释性有限，此处仅示意逻辑）
H = np.load("latent_ae.npy")
labels = np.load("labels_deep.npy")

profile = pd.DataFrame(H, columns=["频率维", "金额维"])
profile["cluster"] = labels
center = profile.groupby("cluster").mean()      # 每簇平均支出画像
print(center.round(3))

# 决策规则: 按频率、金额的均值高低划分四象限
freq_med = center["频率维"].median()
amt_med = center["金额维"].median()
decision = {}
for c, row in center.iterrows():
    hi_f = row["频率维"] > freq_med
    hi_a = row["金额维"] > amt_med
    if hi_f and hi_a:
        decision[c] = "高价值核心客户: VIP 专属权益、新品优先、重点维系"
    elif hi_f and not hi_a:
        decision[c] = "高频低额: 组捆绑销、提升客单价的套装设计"
    elif not hi_f and hi_a:
        decision[c] = "低频高额单: 唤醒召回、精准高价单品推荐"
    else:
        decision[c] = "低活跃低额: 低成本自动化触达、观察是否流失风险"
    print(f"簇{c}: 频率={'高' if hi_f else '低'} 金额={'高' if hi_a else '低'} -> {decision[c]}")

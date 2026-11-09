# step2_make_data.py
# 用途: 生成本实验使用的二分类不平衡数据集，并转为带列名的 DataFrame，便于理解。
# 关键参数:
#   n_samples=20000        总样本数
#   weights=[0.985,0.015]  类别比例: 负类98.5%, 正类1.5%
#   n_informative=10       真正有区分力的特征数
#   flip_y=0.01            1% 标签随机翻转（噪声）
#   random_state=482       保证可复现（按课程规范统一为482）
# 预期输出: 打印 shape 与类别分布; X 为 (20000,20) 数组, y 为 0/1 标签。
# 常见报错与解决:
#   - 无明显报错; 想要更不平衡可把 weights 改成 [0.995,0.005]。
from step1_config import *
import pandas as pd

feature_names = [f"f{i+1}" for i in range(20)]
X, y = make_classification(
    n_samples=20000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    weights=[0.985, 0.015],
    flip_y=0.01,
    random_state=RANDOM_STATE,
)

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

print("数据规模:", df.shape)
print("类别分布:\n", df["label"].value_counts())
print("正类（欺诈）占比: {:.3%}".format(df["label"].mean()))

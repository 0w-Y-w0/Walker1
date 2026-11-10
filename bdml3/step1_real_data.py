# step1_real_data.py
# 用途: 获取真实客户年支出数据 Wholesale Customers，进入"客户分群"场景
# 数据来源: UCI 官方（OpenML 502 号已指向其他数据集且被墙，故改用 UCI 直链）
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 若本地无 CSV 则从 UCI 下载
if not os.path.exists("wholesale.csv"):
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    urllib.request.urlretrieve(url, "wholesale.csv")

df = pd.read_csv("wholesale.csv")
print("原始列:", list(df.columns))

# 去掉 Channel(渠道)/Region(地区) 两个类别列，仅保留 6 类年支出金额
num_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X_real = df[num_cols].values
X_real_s = StandardScaler().fit_transform(X_real)
print("真实客户数据形状:", X_real_s.shape, "特征:", num_cols)
print(X_real_s[:3].round(3))

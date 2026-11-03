import numpy as np 
import pandas as pd 
 
np.random.seed(482) 
n = 100 
# 一元输入 x 在 [-3, 3] 均匀分布 
x = np.random.uniform(-3, 3, n).reshape(-1, 1) 
# 真实关系：非线性 y = x*sin(x) （含周期性，浅层线性模型难以拟合） 
y_true = x * np.sin(x) 
# 加入高斯噪声，模拟真实观测误差 
noise = np.random.normal(0, 0.3, n).reshape(-1, 1) 
y = y_true + noise 
 
# 保存用于两个模型共享 
data = pd.DataFrame({"x": x.ravel(), "y": y.ravel()}) 
data.to_csv("synthetic.csv", index=False)
print("数据已生成，范围 x∈[-3,3]，前 5 行：") 
print(data.head()) 

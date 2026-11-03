import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 

# 解决 matplotlib 中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("synthetic.csv") 
X = df[["x"]].values # 二维 (n,1) 
y = df["y"].values # 一维 (n,) 
 
# 训练线性回归（浅层：单层线性映射） 
model = LinearRegression() 
model.fit(X, y) 
 
# 在密集网格上预测，用于画平滑曲线 
x_grid = np.linspace(-3, 3, 200).reshape(-1, 1) 
y_pred = model.predict(x_grid) 
 
train_mse = mean_squared_error(y, model.predict(X)) 
print(f"线性回归权重 w={model.coef_[0]:.4f}, 偏置 b={model.intercept_:.4f}") 
print(f"训练集 MSE = {train_mse:.4f}") 
 
# 可视化 
plt.figure(figsize=(7, 5)) 
plt.scatter(X, y, s=18, alpha=0.6, label="观测数据(含噪声)") 
plt.plot(x_grid, y_pred, color="red", linewidth=2, label="线性回归拟合") 
plt.title("浅层模型：线性回归") 
plt.xlabel("x"); plt.ylabel("y") 
plt.legend(); plt.grid(alpha=0.3) 
plt.tight_layout(); plt.savefig("linear_fit.png", dpi=120); plt.show() 
print("已保存 linear_fit.png") 

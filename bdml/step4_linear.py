# step4_linear.py
import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy(); y = df["y"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

t0 = time.time()
model = LinearRegression()          # 默认最小二乘（解析解 / 高效求解）
model.fit(X_tr_s, y_tr)
train_time = time.time() - t0

y_pred = model.predict(X_te_s)
mse = mean_squared_error(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
r2 = r2_score(y_te, y_pred)
print(f"训练耗时: {train_time:.2f}s")
print(f"TEST MSE={mse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")
print("学到权重(前3):", np.round(model.coef_[:3], 3), "偏置:", round(model.intercept_, 3))
print("完整权重:", np.round(model.coef_, 3))

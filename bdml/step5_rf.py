# step5_rf.py
import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_parquet("bigdata_reg.parquet")
X = df.drop(columns=["y"]).to_numpy(); y = df["y"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=482)
# 随机森林对尺度不敏感，可跳过标准化；但为加速先用小样本演示
X_tr, y_tr = X_tr[:200_000], y_tr[:200_000]    # 取 20 万训练，平衡速度与效果

t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=100,          # 树的数量 K
    max_features="sqrt",       # 每节点随机选 sqrt(d) 个特征
    n_jobs=-1,                 # 调用所有 CPU 核心
    random_state=482,
    verbose=0,
)
rf.fit(X_tr, y_tr)
train_time = time.time() - t0

y_pred = rf.predict(X_te)
print(f"RF 训练耗时: {train_time:.1f}s")
print(f"MSE={mean_squared_error(y_te, y_pred):.3f} "
      f"MAE={mean_absolute_error(y_te, y_pred):.3f} "
      f"R2={r2_score(y_te, y_pred):.4f}")

# 可选: XGBoost / LightGBM 的 sklearn 接口（进阶）
import lightgbm as lgb
lgbm = lgb.LGBMRegressor(n_estimators=200, n_jobs=-1, random_state=482, verbose=-1)
lgbm.fit(X_tr, y_tr)
print("LGBM R2:", round(r2_score(y_te, lgbm.predict(X_te)), 4))

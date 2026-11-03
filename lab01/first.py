import numpy as np
import pandas as pd

# 设置随机种子，保证结果可复现
np.random.seed(482)

# 模拟 14 天的天气记录
n_days = 14
dates = pd.date_range("2026-03-01", periods=n_days).strftime("%Y-%m-%d")
# 降雨概率：在 0~1 之间随机
rain_prob = np.round(np.random.uniform(0.0, 1.0, n_days), 2)
# 是否阴天：0 或 1，用伯努利分布随机
cloudy = np.random.randint(0, 2, n_days)

df = pd.DataFrame({
    "date": dates,
    "rain_prob": rain_prob,
    "cloudy": cloudy,
})

# 保存为 CSV（index=False 不加行号列）
df.to_csv("weather.csv", index=False, encoding="utf-8-sig")
print("已生成 weather.csv，共", len(df), "条记录")
print(df.head(10))

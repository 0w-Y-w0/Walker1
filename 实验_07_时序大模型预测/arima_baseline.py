# arima_baseline.py
# 步骤3a: Baseline —— ARIMA(经典统计时序模型)
# 用途: 对单通道序列拟合 ARIMA(p,d,q), 输出未来 pred_len 步预测,
#       作为可解释的强基线, 用于对比深度模型。
# 关键参数: order=(2,1,2) 常用小阶组合; 用序列末端 last_n 点拟合
#           (ARIMA 对长序列拟合慢, 截尾是工程惯例, 且越近的数据越相关)。
# 预期输出: 返回长度 pred_len 的 numpy 数组。
# 常见报错与解决:
#   - FitWarning/ConvergenceWarning: 属正常, 可忽略;
#   - 拟合极慢: 调小 last_n 或降低 order;
#   - np.nan 值: ARIMA 对常数段(方差0)会失败, 确认输入有变化。
import warnings
warnings.filterwarnings("ignore")   # statsmodels 的收敛警告过于啰嗦
from statsmodels.tsa.arima.model import ARIMA

def arima_baseline(train_channel, pred_len, order=(2, 1, 2), last_n=500):
    series = train_channel[-last_n:]          # 截尾拟合, 工程惯例
    model = ARIMA(series, order=order)
    fit = model.fit()
    return fit.forecast(steps=pred_len)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("ETTh1.csv")
    data = df.drop(columns=["date"]).values.astype("float32")
    train_ch0 = data[:int(len(data) * 0.7), 0]
    pred = arima_baseline(train_ch0, 24)
    print("ARIMA 未来 24 步预测(通道0 HUFL, 归一化前):")
    print(pred.round(3))

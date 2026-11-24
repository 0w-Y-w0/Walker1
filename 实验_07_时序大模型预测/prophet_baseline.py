# prophet_baseline.py
# 步骤3b: Baseline —— Prophet（可选）
# 用途: 用 Facebook Prophet 建模带季节性的单通道序列, 作为更易解释的业务 baseline。
# 关键参数:
#   daily/weekly_seasonality 开启, yearly 关闭(2 年数据不足以稳健估计年周期);
#   make_future_dataframe(periods=pred_len, freq='h') 小时级预测。
# 预期输出: 打印 Prophet 未来 24 步的 yhat 预测, 形状 (24,)。
# 常见报错与解决:
#   - 安装/编译 prophet 慢或失败(需 cmdstanpy 编译 Stan 模型, Windows 无
#     预编译 wheel 时要 MSVC): 可跳过本 baseline, 仅用 ARIMA 对比, 实验照样成立;
#   - ConvergenceWarning: 属正常现象, 结果仍可用; 可调 interval_width 或增加迭代。
# 本机状态: Python 3.13 无 prophet wheel, 按教程注释选择跳过, 脚本留档。
#   后续若有 Linux 环境或 conda, 直接运行本文件即可。
# 仅当你已 pip install prophet 时运行:
try:
    from prophet import Prophet
    import pandas as pd

    def prophet_baseline(ts_values, pred_len, freq="h"):
        dfp = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=len(ts_values), freq=freq),
                            "y": ts_values})
        m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=pred_len, freq=freq)
        fc = m.predict(future)[["ds", "yhat"]].tail(pred_len)
        return fc["yhat"].values

    if __name__ == "__main__":
        import numpy as np
        df = pd.read_csv("ETTh1.csv")
        data = df.drop(columns=["date"]).values.astype("float32")
        train_ch0 = data[:int(len(data)*0.7), 0]
        pred_prophet = prophet_baseline(train_ch0, 24)
        print("Prophet 预测形状:", pred_prophet.shape)
except ImportError:
    print("prophet 未安装(本机按教程注释跳过该可选 baseline)。")
    print("后续步骤将用 ARIMA 作为可解释 baseline 对比, 实验完整性不受影响。")

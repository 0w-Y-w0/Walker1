# load_etth1.py
# 方案 B: 公开数据集 ETTh1（真实多变量时序, Informer 论文 ETT benchmark）
# 用途: 下载并加载 ETTh1 的 7 通道负荷/油温数据, OT 为重点预测目标。
# 关键参数: file_path 指向 ETTh1.csv。
# 预期输出: 列名 ['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'];
#           形状 (17420, 8); 数值形状 (17420, 7)。
# 常见报错与解决:
#   - 下载被墙 -> 走 ghproxy 代理(已用) 或手动下载;
#   - FileNotFoundError -> 确认脚本与 ETTh1.csv 同目录运行。
import pandas as pd

file_path = "ETTh1.csv"
df = pd.read_csv(file_path)
print("列名:", list(df.columns))
print("形状:", df.shape)
# 取数值列(去掉 date)，并把 OT 作为重点预测目标
values = df.drop(columns=["date"]).values.astype("float32")
print("数值形状:", values.shape)
print("OT 油温统计: min=%.2f max=%.2f mean=%.2f"
      % (values[:, -1].min(), values[:, -1].max(), values[:, -1].mean()))

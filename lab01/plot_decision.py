import pandas as pd 
import matplotlib.pyplot as plt 

# 解决 matplotlib 中文乱码：指定黑体，负号正常显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("weather_with_decision.csv", encoding="utf-8-sig") 
 # 把决策映射为数值，便于画柱状图 
df["dec_num"] = df["decision"].map({"带伞": 1, "不带伞": 0}) 
 
plt.figure(figsize=(10, 4)) 
bars = plt.bar(df["date"], df["dec_num"], color=df["dec_num"].map({1: "tab:blue", 0: 
"lightgray"})) 
plt.yticks([0, 1], ["不带伞", "带伞"]) 
plt.xticks(rotation=45) 
plt.title("每日带伞决策结果") 
plt.xlabel("日期") 
plt.ylabel("决策") 
plt.tight_layout() 
plt.savefig("decision_bar.png", dpi=120) 
plt.show() 
print("决策柱状图已保存为 decision_bar.png") 
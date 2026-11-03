import pandas as pd

# 读取刚才生成的天气数据
df = pd.read_csv("weather.csv", encoding="utf-8-sig")

def decide_umbrella(rain_prob: float, cloudy: int) -> str:
    """
    根据降雨概率与是否阴天决定是否带伞。
    规则：
      - 降雨概率 > 0.6   -> 带伞
      - 0.3 < 降雨概率 <= 0.6 且阴天 -> 带伞
      - 其他 -> 不带伞
    """
    if rain_prob > 0.6:
        return "带伞"
    elif rain_prob > 0.3 and cloudy == 1:
        return "带伞"
    else:
        return "不带伞"

# 对每一天做决策
decisions = []
for _, row in df.iterrows():
    d = decide_umbrella(row["rain_prob"], row["cloudy"])
    decisions.append(d)
    print(f"{row['date']}  降雨概率={row['rain_prob']:.2f}  阴天={int(row['cloudy'])} -> {d}")

# 把决策结果并入原表并保存，便于后续分析与提交
df["decision"] = decisions
df.to_csv("weather_with_decision.csv", index=False, encoding="utf-8-sig")
print("\n带伞天数：", decisions.count("带伞"))
print("不带伞天数：", decisions.count("不带伞"))

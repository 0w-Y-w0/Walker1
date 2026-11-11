# step7_function_call.py
# 4.7 Function Calling——让模型学会"调用工具"
# 用途: 演示 Function Calling: 模型输出"要调用的函数与参数",
#       本地执行函数拿到结果, 再回传模型形成最终决策。
#       这是"LLM + 真实数据"结合的标准模式。
# 关键参数:
#   tools           描述可用函数的 JSON Schema（name/description/parameters）
#   tool_choice="auto"  由模型决定是否调用
# 预期输出: 模型先请求调用 get_exchange_rate(from="USD",to="CNY");
#       本地返回汇率后，模型据此给出是否划算的采购建议。
# 常见报错与解决:
#   - 部分本地模型对 tool_calls 支持弱 -> 可改成"提示词要求模型输出
#     JSON 指定函数名与参数"，本地解析后调用（降级方案）。
#   - 云端返回 tool_calls 为空 -> 任务过简单模型认为不需要;
#     可在 prompt 中强调"必须先查汇率"。
import json
from step1_client import client, model


def get_exchange_rate(frm, to):
    # 真实场景调用汇率 API; 此处固定模拟值演示
    rates = {"USD": 1.0, "CNY": 7.2, "EUR": 0.92}
    return rates[to] / rates[frm]

tools = [{
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "查询两种货币间的汇率, 返回 1 单位 from 货币折合多少 to 货币。",
        "parameters": {
            "type": "object",
            "properties": {
                "frm": {"type": "string", "description": "源货币代码, 如 USD"},
                "to": {"type": "string", "description": "目标货币代码, 如 CNY"}
            },
            "required": ["frm", "to"]
        }
    }
}]

messages = [{"role": "system", "content": "你是跨境采购决策助手。涉及币种换算时必须先调用 get_exchange_rate。"},
            {"role": "user", "content": "美国供应商报价 1000 USD, 国内同类 7200 CNY, 哪个更划算？"}]

# 第一轮: 看模型是否请求调用函数
r1 = client.chat.completions.create(model=model, messages=messages,
                                    tools=tools, tool_choice="auto")
msg = r1.choices[0].message

if msg.tool_calls:
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_exchange_rate(args["frm"], args["to"])
        # 把函数调用与结果追加回对话
        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": json.dumps({"rate": result},
                                               ensure_ascii=False)})
    # 第二轮: 模型基于真实汇率给最终决策
    r2 = client.chat.completions.create(model=model, messages=messages)
    print("最终决策:", r2.choices[0].message.content)
else:
    print("模型未调用函数, 直接回答:", msg.content)

# step4_prompt_paradigms.py
# 4.4 Prompt 工程实践（Zero / Few / CoT 三连击）
# 用途: 对比 Zero-shot、Few-shot、Chain-of-Thought 三种提示范式
#       对同一道"信贷风险初筛"决策任务的输出差异。
# 关键参数:
#   sys_prompt / user_prompt 分别为系统角色与用户任务; CoT 额外要求"先算后判"。
#   temperature=0.0 让对比更公平（尽量确定）。
# 预期输出: 三次打印。Zero-shot 直接给结论; Few-shot 模仿示例格式;
#           CoT 先列计算再下结论，逻辑更清晰。
# 常见报错与解决:
#   - 输出不稳定 -> 本地小模型对 CoT 遵循度弱; 可换更大尺寸或云端模型观察差异。
from step1_client import client, model


def ask(system, user, temp=0.0):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temp, max_tokens=400,
    )
    return r.choices[0].message.content


credit_case = ("申请人: 月收入 8000 元, 月负债 3500 元, 近 6 月逾期 2 次, "
               "信用卡使用率 85%。是否建议批贷? ")

# (1) Zero-shot
z = ask("你是信贷风控助理, 输出是否批贷及一句话理由。", credit_case)

# (2) Few-shot
few = ask(
    "你是信贷风控助理, 参照示例格式输出。",
    "示例: 输入'月入 5000, 负债 2000, 逾期 0, 使用率 30%'→ "
    "输出'建议批贷: 负债比低、无逾期'。\n"
    "现在: " + credit_case
)

# (3) Chain-of-Thought
cot = ask(
    "你是信贷风控助理。请逐步分析负债收入比、逾期次数、信用卡使用率, "
    "再给出批贷结论。",
    credit_case
)

print("=== Zero-shot ===\n", z)
print("\n=== Few-shot ===\n", few)
print("\n=== Chain-of-Thought ===\n", cot)

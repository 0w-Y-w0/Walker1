# vllm_serve.py
# 4.3.3 vLLM（高吞吐服务化，有 GPU 推荐）
# 用途: vLLM 用 PagedAttention 管理显存, 吞吐量远高于 transformers 原生,
#       是生产环境 LLM 服务的主流方案。
# 关键参数:
#   gpu_memory_utilization=0.8  允许 vLLM 占用的显存比例
#   SamplingParams(temperature=0.0, max_tokens=128)
# 注意: vLLM 仅支持 Linux(或 WSL2); Windows 原生无法安装
#       (依赖 Triton/PagedAttention CUDA 内核)。
#       本脚本留档, 在 Linux/WSL2 环境中可直接运行。
# 替代方案: Windows 下已用 FastAPI(app.py) 完成服务化;
#           若需高吞吐, 可在 WSL2 中 pip install vllm 后运行本脚本。
from vllm import LLM, SamplingParams

# 加载模型（自动量化/分片，吞吐远高于 transformers 原生）
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.8)
sampling = SamplingParams(temperature=0.0, max_tokens=128)

prompts = ["你是客服工单决策助手。工单: 用户投诉空调一周不制冷要求退货。请给出类别与建议。"]
outputs = llm.generate(prompts, sampling)
for o in outputs:
    print(o.outputs[0].text)

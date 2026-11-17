# check_env.py
# 用途: 验证 venv06 虚拟环境中微调所需库均可用。
# 预期输出: 各库版本号逐行打印, 无 ImportError 即通过。
# 常见报错与解决:
#   - ModuleNotFoundError -> 脚本运行所用解释器不是 venv06;
#     在 VS Code 右下角切换解释器到 venv06\Scripts\python.exe。
import torch, transformers, peft, datasets, fastapi, uvicorn

print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
print("datasets    ", datasets.__version__)
print("fastapi     ", fastapi.__version__)
print("uvicorn     ", uvicorn.__version__)

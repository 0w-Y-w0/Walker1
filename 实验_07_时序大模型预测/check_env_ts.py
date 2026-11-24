# check_env_ts.py
# 3.3 环境验证
# 用途: 确认版本库版本与 GPU 能否使用, 避免后续运行报错。
# 关键参数: 无。
# 预期输出: 打印各库版本; torch.cuda.is_available() 本地无 GPU 时为 False
#           (本机 RTX 4050 下应为 True)。
# 常见报错与解决:
#   - ImportError -> 解释器不是 venv06, 在 VS Code 右下角切换解释器;
#   - CUDA False 但有显卡 -> torch 装成了 CPU 版, 需重装 cu126 版。
import sys, platform
import numpy, pandas, sklearn, matplotlib, torch, statsmodels
print("Python       :", platform.python_version())
print("numpy        :", numpy.__version__)
print("pandas       :", pandas.__version__)
print("scikit-learn :", sklearn.__version__)
print("matplotlib   :", matplotlib.__version__)
print("torch        :", torch.__version__)
print("statsmodels  :", statsmodels.__version__)
print("CUDA 可用    :", torch.cuda.is_available())

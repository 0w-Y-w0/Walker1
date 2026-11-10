# 用途: 验证实验所需库是否安装成功并打印版本，避免后续 ImportError。
# 关键参数: 无。
# 预期输出: 依次打印各库版本号，且无报错；torch 会打印 CPU/CUDA 后端信息。
import sys, platform
import numpy as np
import pandas as pd
import matplotlib, seaborn, torch
import sklearn
try:
    import imblearn
    imb_ver = imblearn.__version__
except Exception as e:
    imb_ver = f"未安装: {e}"

print("Python         :", sys.version.split()[0])
print("Platform       :", platform.platform())
print("numpy          :", np.__version__)
print("pandas         :", pd.__version__)
print("scikit-learn   :", sklearn.__version__)
print("matplotlib     :", matplotlib.__version__)
print("seaborn        :", seaborn.__version__)
print("torch          :", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
print("imbalanced-learn:", imb_ver)

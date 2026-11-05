# verify_env.py 
import sys, numpy, pandas, sklearn, matplotlib, torch 
print("Python :", sys.version.split()[0]) 
print("numpy :", numpy.__version__) 
print("pandas :", pandas.__version__) 
print("scikit-learn:", sklearn.__version__) 
print("matplotlib :", matplotlib.__version__) 
print("torch :", torch.__version__) 
print("torch CPU OK:", torch.ones(3) @ torch.ones(3)) # 应打印 tensor(3.) 
# 可选 
try: 
 import dask, xgboost, lightgbm 
 print("dask/xgb/lgb:", dask.__version__, xgboost.__version__, 
lightgbm.__version__) 
except ImportError as e: 
 print("可选库未装(可忽略):", e) 

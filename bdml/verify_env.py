# verify_env.py
# 用途: 一次性验证所有核心库版本是否达标
import sys, importlib

required = {
    "numpy": "1.24", "pandas": "2.0", "sklearn": "1.3",
    "matplotlib": "3.7", "scipy": "1.10", "torch": "2.0",
}

def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"MISSING ({e})"

print(f"Python {sys.version.split()[0]}")
for pkg, low in required.items():
    v = ver(pkg)
    print(f"  {pkg:12s} installed={v:12s} required>={low}")

import torch
print("torch CUDA available:", torch.cuda.is_available())

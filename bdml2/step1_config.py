# step1_config.py
# 用途: 集中导入本实验所有库，并设置随机种子与绘图样式，保证可复现。
# 关键参数: RANDOM_STATE = 482（所有随机操作统一种子）。
# 预期输出: 无异常输出，但最后有随境配置完成。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             precision_recall_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

RANDOM_STATE = 482
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")     # 必须在设置字体之前——它会重置 rcParams！
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False
# Windows 下用黑体显示中文; macOS 可用 'Arial Unicode MS', Linux 可用 'WenQuanYi Micro Hei'
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
print("环境配置完成。")

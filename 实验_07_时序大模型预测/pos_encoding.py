# pos_encoding.py
# 4.4.1 位置编码
# 用途: 为 patch 序列加入位置信息(注意力本体对顺序不敏感, 必须显式编码位置)。
# 关键参数: d_model(隐藏维度), max_len(最大 patch 数)。
# 预期输出: 输出形状与输入一致 [B, N, d]; 不同位置的编码向量互不相同。
# 常见报错与解决:
#   - 序列长度超过 max_len -> 索引越界; 建 max_len=5000 一般够用,
#     若 patch 数更大需调大;
#   - 奇偶列错误: 偶数列用 sin、奇数列用 cos, 与 Transformer 论文一致。
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # [max_len, d]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x):
        # x: [B, N, d]
        return x + self.pe[:, :x.size(1), :]

if __name__ == "__main__":
    pe = PositionalEncoding(d_model=64)
    x = torch.zeros(2, 10, 64)          # 模拟 batch=2, patch=10
    y = pe(x)
    print("输出形状:", tuple(y.shape))   # 应为 (2, 10, 64)
    print("位置0与位置1编码是否不同:",
          not torch.allclose(y[0, 0], y[0, 1]))
    print("同一位置跨 batch 是否相同(位置编码与内容无关):",
          torch.allclose(y[0, 5], y[1, 5]))

# patch_tst.py
# 4.4.2 Patch 嵌入与模型主体（PatchTST）
# 用途: 把每个通道的序列切成 patch, 做线性嵌入, 过 Transformer 编码器,
#       最后用线性头一次性输出未来 H 步——PatchTST 式的多变量时序预测器。
# 关键参数:
#   patch_len=16, stride=8   patch 长与步长(可重叠), 96 步 → 11 个 patch
#   d_model=128, n_heads=8   注意力宽度与头数(128/8=16 每头维度)
#   n_layers=3, dropout=0.1
# 预期输出: (8, 96, 2) 的 batch 前向得到 (8, 24, 2) 的预测; 打印参数量。
# 常见报错与解决:
#   - RuntimeError: nhead=8 要求 d_model 能被 n_heads 整除(128/8=16 OK);
#     改 d_model 时务必保证整除;
#   - unfold 维度错误: 输入必须是 3D [B, L, V]; 且 seq_len 足够产生
#     至少 1 个 patch((L-patch_len)/stride + 1 >= 1);
#   - CUDA OOM: 调小 d_model / n_layers / batch_size。
import torch
import torch.nn as nn
from pos_encoding import PositionalEncoding

class PatchTST(nn.Module):
    def __init__(self, n_vars, patch_len, stride, d_model, n_heads,
                 n_layers, dropout, pred_len,
                 use_pos_enc=True, pool="mean"):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = n_vars
        self.pool = pool                      # "mean" 全局池化 / "last" 取最后 patch
        # 每个通道共享一个线性嵌入(通道独立但参数共享)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.use_pos_enc = use_pos_enc        # 消融开关: 验证位置编码重要性
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def _patchify(self, x):
        # x: [B, L, V] -> 通道独立 -> [B*V, L] -> unfold -> [B*V, N, P]
        B, L, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, L)      # [B*V, L]
        patches = x.unfold(1, self.patch_len, self.stride)  # [B*V, N, P]
        return patches, B, V

    def forward(self, x):
        patches, B, V = self._patchify(x)             # [B*V, N, P]
        emb = self.patch_embed(patches)               # [B*V, N, d]
        if self.use_pos_enc:
            emb = self.pos_enc(emb)
        emb = self.dropout(emb)
        enc = self.encoder(emb)                       # [B*V, N, d]
        rep = enc.mean(dim=1) if self.pool == "mean" else enc[:, -1, :]
        out = self.head(rep)                          # [B*V, pred_len]
        out = out.reshape(B, V, -1).permute(0, 2, 1)  # [B, pred_len, V]
        return out

if __name__ == "__main__":
    seq_len, pred_len = 96, 24
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTST(n_vars=2, patch_len=16, stride=8, d_model=128,
                     n_heads=8, n_layers=3, dropout=0.1,
                     pred_len=pred_len).to(device)
    dummy = torch.randn(8, seq_len, 2).to(device)
    out = model(dummy)
    print("模型输出形状:", tuple(out.shape))    # (8, 24, 2)
    total = sum(p.numel() for p in model.parameters())
    print("参数量:", total)
    print("设备:", device)

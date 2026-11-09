# step9_mlp_clf.py
# 用途: 用 PyTorch 实现 MLP 分类器，理解 softmax + 交叉熵在神经网络分类中的应用。
# 关键参数:
#   hidden_dims=[64, 32]  两个隐藏层宽度
#   dropout=0.3           防止过拟合
#   lr=1e-3, batch_size=256, epochs=30
#   weight_class_weight   正类权重更高（与样本数成反比），缓解不平衡
# 预期输出: 打印每 5 轮的训练 loss 与测试 AUC，最终 AUC 应接近 0.93~0.96。
# 常见报错与解决:
#   - CUDA out of memory -> 本实验用 CPU 即可，或减小 batch_size;
#   - loss 不下降 -> 检查 learning_rate 是否过大过小，或特征未标准化;
#   - shape 不匹配 -> CrossEntropyLoss 输入为 (N,2) logits、标签为 (N,) 长整型。
import os
from step1_config import *
from step4_split import X_train_s, X_test_s, y_train, y_test

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class MLP_nn(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 2)]   # 二分类输出 2 个 logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 构造类别权重: 正类权重更高（与样本数成反比）
n_pos = int((y_train.sum())); n_neg = len(y_train) - n_pos
w_pos = len(y_train) / (2 * n_pos)
w_neg = len(y_train) / (2 * n_neg)
class_w = torch.tensor([w_neg, w_pos], dtype=torch.float32)

# 转为张量并构建 DataLoader
X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.long)
X_te_t = torch.tensor(X_test_s, dtype=torch.float32)
y_te_t = torch.tensor(y_test, dtype=torch.long)
train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

device = "cpu"
model = MLP_nn(input_dim=X_train_s.shape[1]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    # 每 5 轮评估一次 AUC
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(X_te_t.to(device)), dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_test, prob)
        print(f"Epoch {epoch:02d} | loss={running_loss/len(train_ds):.4f} | test AUC={auc:.4f}")

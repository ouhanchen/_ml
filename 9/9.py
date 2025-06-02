import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import matplotlib.pyplot as plt # type: ignore

# 設定亂數種子以方便重現結果
torch.manual_seed(42)

# 1. 產生模擬資料：y = 2x + 3 + noise
X = torch.linspace(0, 10, 100).reshape(-1, 1)  # shape: [100, 1]
y = 2 * X + 3 + torch.randn(X.size()) * 0.5   # 加入隨機雜訊

# 2. 建立線性模型 y = wx + b
model = nn.Linear(in_features=1, out_features=1)

# 3. 定義損失函數與優化器
loss_fn = nn.MSELoss()  # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 訓練模型
for epoch in range(200):
    model.train()

    y_pred = model(X)          # 預測
    loss = loss_fn(y_pred, y)  # 計算損失

    optimizer.zero_grad()      # 歸零梯度
    loss.backward()            # 反向傳播
    optimizer.step()           # 更新參數

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss = {loss.item():.4f}")

# 5. 顯示學習到的參數
weight = model.weight.item()
bias = model.bias.item()
print(f"\nLearned parameters:")
print(f"  Weight (w): {weight:.4f}")
print(f"  Bias (b):   {bias:.4f}")

# 6. 預測與視覺化
model.eval()
with torch.no_grad():
    pred = model(X)

plt.figure(figsize=(8, 5))
plt.scatter(X.numpy(), y.numpy(), label='True data', alpha=0.6)
plt.plot(X.numpy(), pred.numpy(), color='red', label='Fitted line')
plt.legend()
plt.title('Linear Regression with PyTorch')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

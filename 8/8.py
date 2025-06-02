import torch # type: ignore

# 初始化變數，requires_grad=True 才能進行自動微分
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

# 超參數：學習率
lr = 0.1

# 梯度下降
for i in range(50):
    # 清除舊的梯度（否則會累加）
    if x.grad is not None:
        x.grad.zero_()
        y.grad.zero_()
        z.grad.zero_()

    # 前向傳播
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 反向傳播
    f.backward()

    # 梯度下降更新
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        z -= lr * z.grad

    print(f"Step {i:02d}: f = {f.item():.6f}, x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")

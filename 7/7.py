from micrograd.engine import Value  # 或使用你貼上的 Value 類別

# 初始化變數（可隨機也可指定）
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

# 超參數：學習率
lr = 0.1

# 梯度下降迭代
for i in range(50):
    # forward pass
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # backward pass
    f.backward()

    # gradient descent update
    x.data -= lr * x.grad
    y.data -= lr * y.grad
    z.data -= lr * z.grad

    # 清除 gradient 以避免累加
    x.grad = 0
    y.grad = 0
    z.grad = 0

    print(f"step {i}: f = {f.data:.6f}, x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")

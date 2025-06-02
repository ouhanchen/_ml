import random

# 目標函數
def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 鄰近點（六個方向）
def get_neighbors(x, y, z, step=0.1):
    return [
        (x + step, y, z),
        (x - step, y, z),
        (x, y + step, z),
        (x, y - step, z),
        (x, y, z + step),
        (x, y, z - step),
    ]

# 爬山（最小化版本）
def hill_climb(start_x, start_y, start_z, step=0.1, max_iter=1000):
    x, y, z = start_x, start_y, start_z
    for _ in range(max_iter):
        current_value = f(x, y, z)
        neighbors = get_neighbors(x, y, z, step)
        next_point = min(neighbors, key=lambda point: f(*point))
        next_value = f(*next_point)

        if next_value < current_value:
            x, y, z = next_point
        else:
            break  # 無法再進步 → 達到局部最小
    return (x, y, z), f(x, y, z)

# 初始隨機位置
start_x = random.uniform(-10, 10)
start_y = random.uniform(-10, 10)
start_z = random.uniform(-10, 10)

# 執行爬山演算法
minimum_point, minimum_value = hill_climb(start_x, start_y, start_z)

print(f"Minimum point: {minimum_point}")
print(f"Function value at minimum: {minimum_value}")

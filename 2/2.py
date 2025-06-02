import random
import math

# 計算兩點距離
def distance(city1, city2):
    return math.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

# 計算整條路徑總距離
def total_distance(path, cities):
    dist = 0
    for i in range(len(path)):
        dist += distance(cities[path[i]], cities[path[(i+1) % len(path)]])  # 回到起點
    return dist

# 產生鄰居：任意兩城市交換
def get_neighbors(path):
    neighbors = []
    for i in range(len(path)):
        for j in range(i+1, len(path)):
            new_path = path.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]
            neighbors.append(new_path)
    return neighbors

# 爬山演算法
def hill_climb(cities, max_iter=1000):
    current_path = list(range(len(cities)))
    random.shuffle(current_path)
    current_distance = total_distance(current_path, cities)

    for _ in range(max_iter):
        neighbors = get_neighbors(current_path)
        better_found = False
        for neighbor in neighbors:
            d = total_distance(neighbor, cities)
            if d < current_distance:
                current_path = neighbor
                current_distance = d
                better_found = True
                break  # 找到更好的就移動（貪婪）
        if not better_found:
            break  # 沒有更好的了

    return current_path, current_distance

# 測試：隨機產生10個城市
num_cities = 10
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

# 執行
best_path, best_dist = hill_climb(cities)

# 顯示結果
print("Best path:", best_path)
print("Shortest distance:", best_dist)

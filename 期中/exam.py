# predict_housing.py

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.datasets import load_boston # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

# 1. 讀取資料
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# 2. 資料分析
print("資料摘要：")
print(df.describe())

# 熱力圖（可選）
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title("Feature Correlation Heatmap")
# plt.show()

# 3. 特徵與目標分離
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# 4. 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. 模型訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 模型預測
y_pred = model.predict(X_test)

# 8. 模型評估
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型評估結果：")
print(f"MAE（平均絕對誤差）：{mae:.2f}")
print(f"MSE（均方誤差）：{mse:.2f}")
print(f"R²（決定係數）：{r2:.2f}")

# 9. 可視化結果
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([0, 50], [0, 50], color='red', linestyle='--')  # 對角線
plt.xlabel("實際房價")
plt.ylabel("預測房價")
plt.title("實際 vs 預測（房價）")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. 使用者自訂輸入預測（模擬一筆資料）
print("\n模擬一筆自訂資料進行預測：")
sample_data = np.array([[0.1, 18.0, 6.2, 0, 0.5, 6.0, 40.0, 4.0, 1, 300.0, 15.0, 390.0, 5.0]])
sample_data_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_data_scaled)
print(f"預測房價（單位：千美元）：{prediction[0]:.2f}")

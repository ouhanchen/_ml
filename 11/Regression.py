# regression_example.py

from sklearn.datasets import load_diabetes # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# 讀取資料集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("均方誤差 (MSE):", mean_squared_error(y_test, y_pred))
print("R² 決定係數:", r2_score(y_test, y_pred))

# gradient_boosting_example.py

from sklearn.datasets import load_wine # type: ignore
from sklearn.ensemble import GradientBoostingClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

# 載入資料集（使用 wine 資料集，非 iris）
data = load_wine()
X, y = data.data, data.target

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立 Gradient Boosting 分類器
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("分類準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# classification_example.py

from sklearn.datasets import load_digits # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

# 讀取手寫數字資料集
digits = load_digits()
X, y = digits.data, digits.target

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("分類準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

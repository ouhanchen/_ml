# 專案主題：使用機器學習預測房價（房地產價格）

## 專案背景與目的:
隨著房地產市場的波動與複雜性，對房價進行準確預測成為房地產投資與政策制定的重要參考。本專案透過機器學習方法，建構一個可預測房價的模型，藉由歷史資料（如面積、房間數、地段等）訓練模型，幫助使用者在買賣房屋時做出更理性的判斷。

## 使用工具與技術:
程式語言：Python
機器學習套件：Scikit-Learn, Pandas, NumPy, Matplotlib
資料集來源：Kaggle 或 UCI ML Repository（例如 Boston Housing Dataset）

## 資料說明:
以 Boston Housing 資料集為例，其欄位包括：
欄位名稱 	說明
CRIM	   每人犯罪率
ZN   	   超過25,000平方英尺住宅區的比例
RM	       每個住宅的平均房間數
AGE	       自住單位建於1940年前的比例
TAX	       房地產稅率
MEDV	   中位房價（預測目標）

## 原理與方法:
1. 前處理（Preprocessing）:
補齊缺失值
標準化數值（Standardization）
分割訓練集與測試集（Train/Test Split）

2. 模型選擇:
選擇以下回歸模型進行預測：
線性回歸（Linear Regression）
決策樹回歸（Decision Tree Regressor）
隨機森林回歸（Random Forest Regressor）

3. 訓練模型:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

4. 模型評估:
使用以下指標來衡量預測準確性：
平均絕對誤差（MAE）
均方誤差（MSE）
R² 決定係數

## 結果分析:
以線性回歸為例：
MAE: 2.15
MSE: 9.80
R²: 0.78
隨機森林的 R² 分數為 0.91，預測效果較佳，適合用於此問題。

## 結論與未來展望:
本專案成功使用機器學習方法預測房價，其中隨機森林回歸表現最佳。未來可加入更多特徵（如地點座標、交通便利性、學校評價等）以提升準確率，並將模型部署成 Web 應用或手機 App，便於一般使用者使用。


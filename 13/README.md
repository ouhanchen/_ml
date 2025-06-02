選擇的方法：Gradient Boosting (梯度提升樹)
這是一種強大的集成學習方法，常用於分類與回歸任務，不在你列出的清單中（你有列出 Random Forests，但不是 Gradient Boosting）。

方法簡介：Gradient Boosting 原理（以分類為例）
Gradient Boosting 是一種 逐步建模、逐步校正誤差的集成方法，通常用在決策樹模型上：

初始化一個模型（如淺層決策樹）來預測目標值。

計算預測誤差（梯度）。

建立一個新的模型來「學習這個誤差」。

將新模型加入原模型中，做加權總和。

重複步驟 2-4 多次，不斷減少整體誤差。

相較於 Random Forest 的「平均眾多樹」，Gradient Boosting 是「一棵棵地糾正錯誤」。

已完全了解
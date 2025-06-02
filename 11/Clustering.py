# clustering_example.py

from sklearn.datasets import make_blobs # type: ignore
from sklearn.cluster import KMeans # type: ignore
import matplotlib.pyplot as plt # type: ignore

# 產生資料（3 群）
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 建立 KMeans 模型
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 視覺化分群結果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, alpha=0.5, marker='X')
plt.title('KMeans Clustering Example')
plt.show()

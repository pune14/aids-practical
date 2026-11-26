import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)
inertia = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertia.append(km.inertia_)
print("Inertia for K=1..8:", np.round(inertia, 2))
plt.plot(range(1, 9), inertia, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
print("Cluster labels:", clusters)
print("Cluster sizes:", np.bincount(clusters))
print("Centroids (scaled):\n", kmeans.cluster_centers_.round(2))
plt.scatter(X[:, 2], X[:, 3], c=clusters)
plt.scatter(
    kmeans.cluster_centers_[:, 2] * X[:, 2].std() + X[:, 2].mean(),
    kmeans.cluster_centers_[:, 3] * X[:, 3].std() + X[:, 3].mean(),
    marker="X", s=200, c="red", label="Centroids"
)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("K-Means Clusters (K = 3)")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
iris = load_iris()
X = iris.data
y = iris.target
X_scaled = StandardScaler().fit_transform(X)
pca_full = PCA().fit(X_scaled)
print("Explained variance ratio:", pca_full.explained_variance_ratio_.round(4))
print("Cumulative variance     :", np.cumsum(pca_full.explained_variance_ratio_).round(4))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Original shape:", X.shape, " Reduced shape:", X_pca.shape)
print("Variance kept by 2 components:", pca.explained_variance_ratio_.sum())
plt.bar(range(1, 5), pca_full.explained_variance_ratio_)
plt.plot(range(1, 5), np.cumsum(pca_full.explained_variance_ratio_), marker="o", color="red")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(True)
plt.show()
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data

# Create DBSCAN object
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit the model and predict clusters
clusters = dbscan.fit_predict(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("DBSCAN Clustering of Iris Dataset")
plt.show()

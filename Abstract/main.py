from typing import List, Tuple

import numpy as np


# Decorator to specify input/output types
def typed(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        output_type = getattr(func, "__output_type__", None)
        if output_type is not None and not isinstance(result, output_type):
            raise TypeError(f"Expected {output_type}, but got {type(result)}")
        return result

    return wrapper


class DBSCAN:
    def __init__(self, eps: float = 0.1, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    @staticmethod
    @typed
    def pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between points in X"""
        n = X.shape[0]
        dist = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                dist[i, j] = dist[j, i] = np.linalg.norm(X[i] - X[j])
        return dist

    @staticmethod
    @typed
    def get_neighbors(distances: np.ndarray, i: int, eps: float) -> List[int]:
        """Get indices of neighbors of point i"""
        return np.where(distances[i] <= eps)[0].tolist()

    @typed
    def expand_cluster(
        self,
        distances: np.ndarray,
        cluster_labels: np.ndarray,
        i: int,
        neighbors: List[int],
        cluster_id: int,
        eps: float,
        min_samples: int,
    ):
        """Expand cluster from seed point i"""
        cluster_labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            k = neighbors[j]
            if cluster_labels[k] == -1:  # unclassified point
                cluster_labels[k] = cluster_id
            elif cluster_labels[k] == 0:  # noise point
                cluster_labels[k] = cluster_id
                k_neighbors = self.get_neighbors(distances, k, eps)
                if len(k_neighbors) >= min_samples:
                    neighbors += k_neighbors
            j += 1

    @typed
    def assign_noise(self, distances: np.ndarray, cluster_labels: np.ndarray):
        """Assign noise points to cluster -1"""
        cluster_labels[cluster_labels == 0] = -1

    @typed
    def set_params(self, **params):
        """Set parameters for the DBSCAN model"""
        if "eps" in params:
            self.eps = params["eps"]
        if "min_samples" in params:
            self.min_samples = params["min_samples"]

    @typed
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN model to data X"""
        n = X.shape[0]
        distances = self.pairwise_distances(X)
        cluster_labels = np.zeros(n, dtype=int)  # initialize as noise points

        cluster_id = 0
        for i in range(n):
            if cluster_labels[i] != 0:
                continue  # point is already in a cluster

            neighbors = self.get_neighbors(distances, i, self.eps)
            if len(neighbors) < self.min_samples:
                cluster_labels[i] = -1  # mark as noise point
            else:  # expand cluster from seed point i
                cluster_id += 1
                self.expand_cluster(distances, cluster_labels, i, neighbors, cluster_id, self.eps, self.min_samples)

        self.assign_noise(distances, cluster_labels)
        return cluster_labels


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs, make_hastie_10_2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Load iris dataset
iris = load_iris()
X_iris = iris.data

# Generate blob dataset
X_blob, y_blob = make_blobs(n_samples=1000, centers=5, random_state=42)

# Generate hastie dataset
X_hastie, y_hastie = make_hastie_10_2(n_samples=1000, random_state=42)

# Define pipeline with PCA and DBSCAN
pipeline = Pipeline([("reduce_dim", PCA()), ("cluster", DBSCAN())])

# Define hyperparameters grid for GridSearchCV
param_grid = {
    "reduce_dim__n_components": [2, 3],  # number of principal components to keep
    "cluster__eps": [0.4, 0.5],
    "cluster__min_samples": [5, 10],
}

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_iris, iris.target, test_size=0.2, random_state=42)

# Define silhouette scorer for GridSearchCV
scorer = silhouette_score

# Run GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring=scorer)
grid_search.fit(X_train, y_train)

# Get best hyperparameters and evaluate on test set
best_params = grid_search.best_params_
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)
labels = pipeline.predict(X_test)
silhouette_avg = silhouette_score(X_test, labels)
print("Best hyperparameters:", best_params)
print("Silhouette score:", silhouette_avg)


# Plot results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].scatter(X_iris[:, 0], X_iris[:, 1], c=labels_iris)
axs[0].set_title("Iris dataset")

axs[1].scatter(X_blob[:, 0], X_blob[:, 1], c=labels_blob)
axs[1].set_title("Blob dataset")

axs[2].scatter(X_hastie[:, 0], X_hastie[:, 1], c=labels_hastie)
axs[2].set_title("Hastie dataset")

plt.show()

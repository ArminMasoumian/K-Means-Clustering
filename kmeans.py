# Author: Armin Masoumian (masoumian.armin@gmail.com)

import numpy as np

class KMeans:
    def __init__(self, n_clusters=2, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        # initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]
        
        for i in range(self.max_iter):
            # assign each data point to the closest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # update centroids
            for j in range(self.n_clusters):
                self.centroids[j] = X[labels == j].mean(axis=0)
        
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

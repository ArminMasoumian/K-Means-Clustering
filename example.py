import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

# generate toy data
np.random.seed(0)
X = np.concatenate([np.random.randn(10, 2) + [0, 10], 
                    np.random.randn(10, 2) + [10, 0], 
                    np.random.randn(10, 2) + [10, 10]])

# run K-Means algorithm
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)

# plot results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()

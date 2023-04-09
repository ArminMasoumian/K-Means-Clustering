import numpy as np
from kmeans import KMeans

# load your dataset here
X = np.loadtxt('your_dataset.csv')

# run K-Means algorithm
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.predict(X)

# print the cluster labels for each data point
print(labels)

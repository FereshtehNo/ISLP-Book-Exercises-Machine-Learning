import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Flame dataset from Spiral.txt
flame_data = np.loadtxt("D:/Phd-classes/Machine-learning/HW4/Spiral.txt")

# (i) K-means
# Specify the range of cluster numbers to try
cluster_range = range(2, 10)

# Try each cluster number and store the results
kmeans_results = {}
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_results[n_clusters] = kmeans.fit_predict(flame_data)

# (ii) Hierarchical clustering (average linkage)
# Compute the linkage matrix
Z_avg = linkage(flame_data, method='average')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z_avg)
plt.title('Hierarchical Clustering (Average Linkage)')
plt.show()

# (iii) Hierarchical clustering (single linkage)
# Compute the linkage matrix
Z_single = linkage(flame_data, method='single')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z_single)
plt.title('Hierarchical Clustering (Single Linkage)')
plt.show()

# (iv) Hierarchical clustering (complete linkage)
# Compute the linkage matrix
Z_complete = linkage(flame_data, method='complete')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z_complete)
plt.title('Hierarchical Clustering (Complete Linkage)')
plt.show()

# (v) Spectral clustering
# Specify the range of cluster numbers to try
cluster_range_spectral = range(2, 10)

# Try each cluster number and store the results
spectral_results = {}
for n_clusters in cluster_range_spectral:
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    spectral_results[n_clusters] = spectral.fit_predict(flame_data)

# You can now inspect the results and choose the best cluster number for each method

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the dataset
file_path = 'D:/Phd-classes/Machine-learning/HW4/USArrests.csv'
df = pd.read_csv(file_path, index_col=0)

# Perform hierarchical clustering
linked = linkage(df, 'ward')

# Plot the dendrogram to visually inspect the clusters
plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('States')
plt.ylabel('Distance')
plt.show()

# Cut the dendrogram at a height to form three distinct clusters
num_clusters = 3
clusters = fcluster(linked, num_clusters, criterion='maxclust')

# Print the states and their corresponding clusters
for state, cluster in zip(df.index, clusters):
    print(f"{state}: Cluster {cluster}")

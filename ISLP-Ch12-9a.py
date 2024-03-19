import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import matplotlib.pyplot as plt

# Set the working directory
os.chdir("D:/Phd-classes/Machine-learning/HW4")

# Load the dataset
data = pd.read_csv("USArrests.csv", index_col=0)

# Standardize the data (optional, but recommended for Euclidean distance)
scaled_data = StandardScaler().fit_transform(data)

# Perform hierarchical clustering with complete linkage and Euclidean distance
hc_complete = linkage(scaled_data, method="complete")

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(hc_complete, labels=data.index, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering - Complete Linkage")
plt.xlabel("States")
plt.show()

# Cut the dendrogram to create clusters (you can adjust the number of clusters as needed)
clusters = cut_tree(hc_complete, n_clusters=3)

# Print the cluster assignments
print(clusters)

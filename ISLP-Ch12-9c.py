import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load the data
file_path = 'D:/Phd-classes/Machine-learning/HW4/USArrests.csv'
data = pd.read_csv(file_path, index_col=0)

# Standardize the data (scaling to have standard deviation one)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Hierarchical clustering with complete linkage and Euclidean distance
linkage_matrix = linkage(scaled_data, method='complete', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=data.index, orientation='top', distance_sort='descending', leaf_font_size=10)
plt.title('Hierarchical Clustering of US States')
plt.xlabel('States')
plt.ylabel('Distance')
plt.show()

from sklearn.cluster import KMeans
import pandas as pd

# Perform K-means clustering with K=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Create a DataFrame with true class labels and clustering labels
comparison_df = pd.DataFrame({'True Class': np.repeat(range(3), n_obs_per_class), 'Cluster': clusters})

# Print crosstab for comparison
cross_tab = pd.crosstab(comparison_df['True Class'], comparison_df['Cluster'])
print(cross_tab)
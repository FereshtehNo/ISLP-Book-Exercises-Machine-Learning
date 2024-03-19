# Perform K-means clustering with K=2
kmeans_2 = KMeans(n_clusters=2, random_state=42)
clusters_2 = kmeans_2.fit_predict(data)

# Create a DataFrame with true class labels and clustering labels for K=2
comparison_df_2 = pd.DataFrame({'True Class': np.repeat(range(3), n_obs_per_class), 'Cluster': clusters_2})

# Print crosstab for comparison with K=2
cross_tab_2 = pd.crosstab(comparison_df_2['True Class'], comparison_df_2['Cluster'])
print(cross_tab_2)

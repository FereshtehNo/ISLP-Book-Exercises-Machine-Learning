# Perform K-means clustering with K=4
kmeans_4 = KMeans(n_clusters=4, random_state=42)
clusters_4 = kmeans_4.fit_predict(data)

# Create a DataFrame with true class labels and clustering labels for K=4
comparison_df_4 = pd.DataFrame({'True Class': np.repeat(range(3), n_obs_per_class), 'Cluster': clusters_4})

# Print crosstab for comparison with K=4
cross_tab_4 = pd.crosstab(comparison_df_4['True Class'], comparison_df_4['Cluster'])
print(cross_tab_4)

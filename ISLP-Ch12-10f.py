# Extract the first two principal components
pc_scores = pca_df[['PC1', 'PC2']]

# Perform K-means clustering with K=3 on the principal components
kmeans_pca = KMeans(n_clusters=3, random_state=42)
clusters_pca = kmeans_pca.fit_predict(pc_scores)

# Create a DataFrame with true class labels and clustering labels for PCA
comparison_df_pca = pd.DataFrame({'True Class': np.repeat(range(3), n_obs_per_class), 'Cluster': clusters_pca})

# Print crosstab for comparison with PCA
cross_tab_pca = pd.crosstab(comparison_df_pca['True Class'], comparison_df_pca['Cluster'])
print(cross_tab_pca)

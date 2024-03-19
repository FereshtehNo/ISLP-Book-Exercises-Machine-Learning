from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Perform K-means clustering with K=3 on the scaled data
kmeans_scaled = KMeans(n_clusters=3, random_state=42)
clusters_scaled = kmeans_scaled.fit_predict(scaled_data)

# Create a DataFrame with true class labels and clustering labels for scaled data
comparison_df_scaled = pd.DataFrame({'True Class': np.repeat(range(3), n_obs_per_class), 'Cluster': clusters_scaled})

# Print crosstab for comparison with scaled data
cross_tab_scaled = pd.crosstab(comparison_df_scaled['True Class'], comparison_df_scaled['Cluster'])
print(cross_tab_scaled)

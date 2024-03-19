from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

# Create a DataFrame with the principal components and class labels
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Class'] = np.repeat(range(3), n_obs_per_class)

# Plot the first two principal component score vectors
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Class'], cmap='viridis')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

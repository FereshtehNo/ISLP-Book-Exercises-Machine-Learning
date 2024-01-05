import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100

# Generate predictor X with a normal distribution (mean=0, std=1)
X = np.random.normal(0, 1, n)

# Generate noise vector epsilon with a normal distribution (mean=0, std=1)
epsilon = np.random.normal(0, 1, n)

# Print the first few values of X and epsilon for inspection
print("Predictor X:")
print(X[:10])  # Print the first 10 values of X
print("\nNoise vector epsilon:")
print(epsilon[:10])  # Print the first 10 values of epsilon

import numpy as np

# Set the random seed for reproducibility
np.random.seed(0)

# Define the dimensions
n = 1000  # Number of observations
p = 20    # Number of features

# Generate X matrix with random values from a standard normal distribution
X = np.random.randn(n, p)

# Generate beta vector with random coefficients, some of which are exactly zero
beta = np.zeros(p)
# Set non-zero coefficients for specific features
beta[0] = 1.0
beta[5] = -0.5

# Generate error term epsilon with random noise
epsilon = np.random.normal(0, 1, n)

# Calculate the response vector Y using the linear model
Y = np.dot(X, beta) + epsilon

# Split the dataset into training and test sets (100 observations for training, 900 observations for testing)
X_train = X[:100, :]
Y_train = Y[:100]
X_test = X[100:, :]
Y_test = Y[100:]

# Ensure the dimensions of X_test and Y_test match
assert X_test.shape[0] == Y_test.shape[0] == 900

# Display the dimensions of the training and test sets
print(f"Training set dimensions: X_train({X_train.shape}), Y_train({Y_train.shape})")
print(f"Test set dimensions: X_test({X_test.shape}), Y_test({Y_test.shape})")

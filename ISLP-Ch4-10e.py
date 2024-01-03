import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=0)

# Initialize variables to track minimum MSE and the corresponding model size
min_mse = float('inf')
best_model_size = None
mse_values = []

# Perform feature selection with varying model sizes
for model_size in range(1, p + 1):
    model = LinearRegression()
    model.fit(X_train[:, :model_size], Y_train)
    Y_test_pred = model.predict(X_test[:, :model_size])
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    mse_values.append(test_mse)

    # Update the minimum MSE and the corresponding model size
    if test_mse < min_mse:
        min_mse = test_mse
        best_model_size = model_size

# Plot the test set MSE associated with different model sizes
model_sizes = range(1, p + 1)
plt.plot(model_sizes, mse_values, marker='o')
plt.xlabel('Model Size (Number of Features)')
plt.ylabel('Test MSE')
plt.title('Model Size vs. Test MSE')
plt.grid(True)
plt.show()

# Print the model size that minimizes the test MSE
print("Best Model Size (minimizing Test MSE):", best_model_size)
print("Corresponding Test MSE:", min_mse)

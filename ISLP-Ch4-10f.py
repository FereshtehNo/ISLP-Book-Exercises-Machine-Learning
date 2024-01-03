import numpy as np
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

# True coefficient values
true_beta = np.zeros(p)
true_beta[0] = 1.0
true_beta[5] = -0.5

# Generate error term epsilon with random noise
epsilon = np.random.normal(0, 1, n)

# Calculate the response vector Y using the true linear model
Y = np.dot(X, true_beta) + epsilon

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=0)

# Initialize variables to track minimum MSE and the corresponding model size
min_mse = float('inf')
best_model = None
mse_values = []

# Perform feature selection with varying model sizes
for model_size in range(1, p + 1):
    model = LinearRegression()
    model.fit(X_train[:, :model_size], Y_train)
    Y_test_pred = model.predict(X_test[:, :model_size])
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    mse_values.append(test_mse)

    # Update the minimum MSE and the corresponding model
    if test_mse < min_mse:
        min_mse = test_mse
        best_model = model

# Print the true coefficients and estimated coefficients of the best model
print("True Coefficients: ", true_beta)
print("Estimated Coefficients (Best Model): ", best_model.coef_)
print("Test Set MSE (Best Model): ", min_mse)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict on the test data
Y_pred = model.predict(X_test)

# Calculate the test error (mean squared error)
test_error = mean_squared_error(Y_test, Y_pred)
print(f"Test Error (MSE): {test_error:.2f}")

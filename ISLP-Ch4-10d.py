import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
np.random.seed(0)

# Define the dimensions
n = 100  # Number of observations for training
p = 20   # Number of features

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

# Perform forward selection for feature selection
selected_features = []
test_mse_values = []

for i in range(p):
    best_mse = float("inf")
    best_feature = None

    for feature in range(p):
        if feature not in selected_features:
            model = LinearRegression()
            current_features = selected_features + [feature]
            model.fit(X_train[:, current_features], Y_train)
            Y_pred = model.predict(X_test[:, current_features])
            mse = mean_squared_error(Y_test, Y_pred)

            if mse < best_mse:
                best_mse = mse
                best_feature = feature

    selected_features.append(best_feature)
    test_mse_values.append(best_mse)

# Plot the test set MSE associated with the best model of each size
num_features = range(1, p + 1)
plt.plot(num_features, test_mse_values, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Test MSE')
plt.title('Forward Selection - Test MSE')
plt.grid(True)
plt.show()

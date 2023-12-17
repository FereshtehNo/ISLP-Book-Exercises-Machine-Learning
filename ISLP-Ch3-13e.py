import numpy as np
from sklearn.linear_model import LinearRegression

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'x' with 100 observations from N(0, 1) distribution
x = np.random.normal(0, 1, 100)

# Generate a vector 'eps' with 100 observations from N(0, 0.25) distribution
eps = np.random.normal(0, 0.25, 100)

# Calculate 'y' according to the specified linear model
y = -1 + 0.5 * x + eps

# Create and fit a linear regression model
model = LinearRegression()
x = x.reshape(-1, 1)  # Reshape 'x' to a 2D array for scikit-learn
model.fit(x, y)

# Get the estimated coefficients (βˆ0 and βˆ1)
beta_hat_0 = model.intercept_
beta_hat_1 = model.coef_[0]

# Print the estimated coefficients
print("Estimated Beta Hat 0:", beta_hat_0)
print("Estimated Beta Hat 1:", beta_hat_1)

# Comment on the model
print("The least squares linear model obtained is: y = {:.4f} + {:.4f}x".format(beta_hat_0, beta_hat_1))

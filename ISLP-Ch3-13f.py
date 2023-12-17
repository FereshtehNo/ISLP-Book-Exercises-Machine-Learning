import numpy as np
import matplotlib.pyplot as plt
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

# Create the scatterplot
plt.scatter(x, y, label='Data Points', alpha=0.7)  # Scatterplot of data points

# Plot the least squares regression line
plt.plot(x, model.predict(x), color='red', label='Least Squares Line')

# Plot the population regression line (true line)
plt.plot(x, -1 + 0.5 * x, color='green', linestyle='--', label='Population Regression Line')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatterplot with Regression Lines')
plt.legend()

# Show the plot
plt.show()

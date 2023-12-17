import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'x' with 100 observations from N(0, 1) distribution
x = np.random.normal(0, 1, 100)

# Generate a vector 'eps' with 100 observations from N(0, 0.25) distribution
eps = np.random.normal(0, 0.25, 100)

# Calculate 'y' according to the specified linear model
y = -1 + 0.5 * x + eps

# Create a polynomial regression model with degree 2
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x.reshape(-1, 1))

model = LinearRegression()
model.fit(x_poly, y)

# Generate points for the polynomial regression curve
x_range = np.linspace(min(x), max(x), 100)
x_range_poly = poly.transform(x_range.reshape(-1, 1))
y_pred = model.predict(x_range_poly)

# Plot the data points
plt.scatter(x, y, label='Data Points', alpha=0.7)

# Plot the polynomial regression curve
plt.plot(x_range, y_pred, color='red', label='Polynomial Regression (Degree 2)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression (Degree 2) vs. Data Points')
plt.legend()

# Show the plot
plt.show()

# Calculate the R-squared value to assess model fit
from sklearn.metrics import r2_score

y_pred = model.predict(x_poly)
r_squared = r2_score(y, y_pred)

print("R-squared value for the polynomial regression model:", r_squared)

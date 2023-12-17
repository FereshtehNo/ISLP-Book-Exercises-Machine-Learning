import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'x' with 100 observations from N(0, 1) distribution
x = np.random.normal(0, 1, 100)

# Generate a vector 'eps' with 100 observations from N(0, 0.25) distribution
eps = np.random.normal(0, 0.25, 100)

# Calculate 'y' according to the specified linear model
y = -1 + 0.5 * x + eps

# Create a scatterplot
plt.scatter(x, y, label='Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatterplot of x and y')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

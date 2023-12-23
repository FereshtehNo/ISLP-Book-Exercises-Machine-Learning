import numpy as np
import matplotlib.pyplot as plt

# Simulated data
rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = x - 2 * x**2 + rng.normal(size=100)

# Create a scatterplot
plt.scatter(x, y, c='blue', label='Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatterplot of X vs. Y')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

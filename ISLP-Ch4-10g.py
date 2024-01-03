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

# Initialize variables to track squared differences for different values of r
squared_diffs = np.zeros((p, p))

# Perform feature selection with varying model sizes
for model_size in range(1, p + 1):
    model = LinearRegression()
    model.fit(X_train[:, :model_size], Y_train)
    estimated_beta = model.coef_
    squared_diff = np.square(true_beta[:model_size] - estimated_beta)
    squared_diffs[model_size - 1, :model_size] = squared_diff

# Create a plot displaying the squared differences for different values of r
model_sizes = range(1, p + 1)
for j in range(p):
    plt.plot(model_sizes, squared_diffs[:, j], label=f'β{str(j+1)}')

plt.xlabel('Model Size (Number of Features)')
plt.ylabel('Squared Difference (βj - β^j)^2')
plt.title('Squared Differences between True and Estimated Coefficients')
plt.legend(loc='upper right', title='Coefficients')
plt.grid(True)
plt.show()

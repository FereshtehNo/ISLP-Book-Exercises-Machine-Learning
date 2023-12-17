import numpy as np

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'x' with 100 observations from N(0, 1) distribution
x = np.random.normal(0, 1, 100)

# Generate a vector 'eps' with 100 observations from N(0, 0.25) distribution
eps = np.random.normal(0, 0.25, 100)

# Calculate 'y' according to the specified linear model
y = -1 + 0.5 * x + eps

# Print the length of 'y'
print("Length of y:", len(y))

# Print the values of β0 and β1
beta0 = -1
beta1 = 0.5
print("Beta0:", beta0)
print("Beta1:", beta1)
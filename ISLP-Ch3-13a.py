import numpy as np

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'x' with 100 observations from N(0,1) distribution
x = np.random.normal(0, 1, 100)

# Print the first few elements of the vector for verification
print(x[:5])
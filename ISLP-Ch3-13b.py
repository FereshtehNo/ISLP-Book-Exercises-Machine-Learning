import numpy as np

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'eps' with 100 observations from N(0, 0.25) distribution
eps = np.random.normal(0, 0.25, 100)

# Print the first few elements of the vector for verification
print(eps[:5])
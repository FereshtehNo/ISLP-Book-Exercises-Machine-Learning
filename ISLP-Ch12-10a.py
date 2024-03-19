import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Number of observations in each class
n_obs_per_class = 20

# Number of variables
n_variables = 50

# Generate simulated data
data_list = []

for i in range(3):
    # Mean shift for each class
    mean_shift = np.random.uniform(-5, 5, size=n_variables)
    
    # Generate data for each class
    class_data = np.random.normal(loc=mean_shift, scale=1, size=(n_obs_per_class, n_variables))
    
    # Append data to the list
    data_list.append(pd.DataFrame(class_data, columns=[f'Variable_{j+1}' for j in range(n_variables)]))

# Concatenate dataframes in the list
data = pd.concat(data_list, ignore_index=True)

# Display the first few rows of the generated data
print(data.head())

# Plot the first two variables for visualization
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=np.repeat(range(3), n_obs_per_class), cmap='viridis')
plt.title('Simulated Data with Three Classes')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.show()

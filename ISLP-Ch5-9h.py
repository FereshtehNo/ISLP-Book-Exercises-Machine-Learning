import pandas as pd
import numpy as np

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Set the number of bootstrap samples
num_bootstrap_samples = 1000

# Initialize an array to store the resampled tenth percentiles
resampled_tenth_percentiles = []

# Calculate the number of observations
num_observations = len(boston_data)

# Perform bootstrapping
for _ in range(num_bootstrap_samples):
    # Randomly sample with replacement from the data
    resampled_data = boston_data.sample(n=num_observations, replace=True)
    
    # Calculate the tenth percentile for the resampled data
    resampled_tenth_percentile = np.percentile(resampled_data['medv'], 10)
    
    # Append the resampled tenth percentile to the array
    resampled_tenth_percentiles.append(resampled_tenth_percentile)

# Calculate the standard error of µ̂0.1 using the bootstrap
bootstrap_se_tenth_percentile = np.std(resampled_tenth_percentiles)

# Print the estimated standard error of µ̂0.1
print("Estimated standard error of µ̂0.1 (using bootstrap):", bootstrap_se_tenth_percentile)

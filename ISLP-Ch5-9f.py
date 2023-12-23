import pandas as pd
import numpy as np

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Set the number of bootstrap samples
num_bootstrap_samples = 1000

# Initialize an array to store the resampled medians
resampled_medians = []

# Calculate the number of observations
num_observations = len(boston_data)

# Perform bootstrapping
for _ in range(num_bootstrap_samples):
    # Randomly sample with replacement from the data
    resampled_data = boston_data.sample(n=num_observations, replace=True)
    
    # Calculate the sample median for the resampled data
    resampled_median = resampled_data['medv'].median()
    
    # Append the resampled median to the array
    resampled_medians.append(resampled_median)

# Calculate the standard error of the median using the bootstrap
bootstrap_se_median = np.std(resampled_medians)

# Print the estimated standard error of the median
print("Estimated standard error of the median (using bootstrap):", bootstrap_se_median)


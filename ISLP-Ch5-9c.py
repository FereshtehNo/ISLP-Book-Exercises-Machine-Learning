import pandas as pd
import numpy as np

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Set the number of bootstrap samples
num_bootstrap_samples = 1000

# Initialize an array to store the resampled sample means
resampled_sample_means = []

# Calculate the number of observations
num_observations = len(boston_data)

# Perform bootstrapping
for _ in range(num_bootstrap_samples):
    # Randomly sample with replacement from the data
    resampled_data = boston_data.sample(n=num_observations, replace=True)
    
    # Calculate the sample mean for the resampled data
    resampled_sample_mean = resampled_data['medv'].mean()
    
    # Append the resampled sample mean to the array
    resampled_sample_means.append(resampled_sample_mean)

# Calculate the standard error of µ̂ using the bootstrap
bootstrap_se_sample_mean_medv = np.std(resampled_sample_means)

# Print the estimated standard error using the bootstrap
print("Estimated standard error of µ̂ using the bootstrap:", bootstrap_se_sample_mean_medv)

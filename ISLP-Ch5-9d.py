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

# Calculate the lower and upper bounds of the 95% confidence interval using the bootstrap
lower_bound = np.percentile(resampled_sample_means, 2.5)
upper_bound = np.percentile(resampled_sample_means, 97.5)

# Calculate the sample mean and sample standard deviation
sample_mean = boston_data['medv'].mean()
sample_std = boston_data['medv'].std()

# Calculate the standard error of the sample mean
se_sample_mean = sample_std / np.sqrt(num_observations)

# Calculate the confidence interval using the standard error
z_value = 1.96  # for a 95% confidence interval
lower_bound_se = sample_mean - z_value * (sample_std / np.sqrt(num_observations))
upper_bound_se = sample_mean + z_value * (sample_std / np.sqrt(num_observations))

# Calculate the confidence interval using the two standard error rule
lower_bound_2se = sample_mean - 2 * se_sample_mean
upper_bound_2se = sample_mean + 2 * se_sample_mean

# Print the results
print("95% Confidence Interval using Bootstrap: ({:.2f}, {:.2f})".format(lower_bound, upper_bound))
print("95% Confidence Interval using Sample Standard Deviation and SE: ({:.2f}, {:.2f})".format(lower_bound_se, upper_bound_se))
print("95% Confidence Interval using Two Standard Error Rule: ({:.2f}, {:.2f})".format(lower_bound_2se, upper_bound_2se))

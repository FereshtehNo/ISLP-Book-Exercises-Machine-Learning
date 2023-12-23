import pandas as pd
import numpy as np

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Calculate the sample mean of 'medv'
sample_mean_medv = boston_data['medv'].mean()

# Calculate the sample standard deviation of 'medv'
sample_std_medv = boston_data['medv'].std()

# Calculate the number of observations
num_observations = len(boston_data)

# Calculate the standard error of the sample mean
se_sample_mean_medv = sample_std_medv / np.sqrt(num_observations)

# Print the estimated standard error
print("Estimated standard error of µ̂ for 'medv':", se_sample_mean_medv)

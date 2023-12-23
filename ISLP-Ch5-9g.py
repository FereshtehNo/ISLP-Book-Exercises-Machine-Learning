import pandas as pd
import numpy as np

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Calculate the tenth percentile of 'medv'
tenth_percentile_medv = np.percentile(boston_data['medv'], 10)

# Print the estimated tenth percentile
print("Estimated tenth percentile (µ̂0.1) of 'medv':", tenth_percentile_medv)

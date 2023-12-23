import pandas as pd

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Calculate the sample mean of 'medv'
sample_mean_medv = boston_data['medv'].mean()

# Print the estimated sample mean
print("Estimated population mean (sample mean) of 'medv':", sample_mean_medv)

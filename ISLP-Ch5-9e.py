import pandas as pd

# Specify the file path to your Boston data
file_path = 'D:/Phd-classes/Machine-learning/HW2/Boston.csv'

# Load the Boston housing data set
boston_data = pd.read_csv(file_path)

# Calculate the sample median of 'medv'
sample_median_medv = boston_data['medv'].median()

# Print the estimated sample median
print("Estimated population median (sample median) of 'medv':", sample_median_medv)

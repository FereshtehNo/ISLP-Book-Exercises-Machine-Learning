import pandas as pd

# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Find the suburb with the lowest median value of owner-occupied homes
lowest_medv_suburb = boston[boston['medv'] == boston['medv'].min()]

# Display the values of all predictors for that suburb
print("Suburb with the Lowest Median Value of Owner-Occupied Homes:")
print(lowest_medv_suburb)

# Calculate the overall ranges for the predictors
ranges = boston.describe().loc[['min', 'max']]

# Display the overall ranges for the predictors
print("\nOverall Ranges for Predictors:")
print(ranges)
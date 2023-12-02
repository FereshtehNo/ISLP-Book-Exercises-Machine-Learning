import pandas as pd

# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Calculate the correlations between CRIM (first column, index 0) and all other predictors
correlations = boston.corr()[boston.columns[0]]

# Display the correlations in descending order
correlations = correlations.sort_values(ascending=False)
print(correlations)
import pandas as pd

# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Get the number of rows and columns in the DataFrame
num_rows, num_columns = boston.shape

# Display the number of rows and columns
print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")
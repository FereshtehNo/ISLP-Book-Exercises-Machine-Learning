import pandas as pd

# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Count the number of suburbs adjacent to the Charles River
suburbs_on_charles = boston[boston['chas'] == 1]

# Get the count of suburbs on the Charles River
num_suburbs_on_charles = suburbs_on_charles.shape[0]

print("Number of suburbs adjacent to the Charles River:", num_suburbs_on_charles)
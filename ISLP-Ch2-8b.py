import pandas as pd

# Set the correct file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\College.csv'

# Read the data into a DataFrame
college = pd.read_csv(file_path)

# Set the first column as the index and rename it 'College'
college = college.rename(columns={'Unnamed: 0': 'College'}).set_index('College')
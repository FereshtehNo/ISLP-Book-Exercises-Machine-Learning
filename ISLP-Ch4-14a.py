import pandas as pd

# Load your data from the specified location
data_path = r'D:\Phd-classes\Machine-learning\HW3\Auto.csv'
Auto = pd.read_csv(data_path)

# Calculate the median of the 'mpg' column
mpg_median = Auto['mpg'].median()

# Create the 'mpg01' column with 1s and 0s based on the median
Auto['mpg01'] = (Auto['mpg'] > mpg_median).astype(int)

# Display the modified data frame
print(Auto)

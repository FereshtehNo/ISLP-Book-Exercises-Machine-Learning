import pandas as pd

# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Calculate the range of each predictor
crime_range = boston['crim'].max() - boston['crim'].min()
tax_range = boston['tax'].max() - boston['tax'].min()
pt_ratio_range = boston['ptratio'].max() - boston['ptratio'].min()

# Display the range of each predictor
print("Range of Crime Rates: {:.2f}".format(crime_range))
print("Range of Tax Rates: {:.2f}".format(tax_range))
print("Range of Pupil-Teacher Ratios: {:.2f}".format(pt_ratio_range))
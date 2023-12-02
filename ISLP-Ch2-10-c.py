import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Create pairwise scatterplots
sns.set(style="ticks")
sns.pairplot(boston)
plt.show()
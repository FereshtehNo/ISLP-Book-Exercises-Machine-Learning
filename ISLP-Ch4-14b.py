import matplotlib.pyplot as plt
import seaborn as sns

# Create scatterplots to visualize the association between 'mpg01' and other features
plt.figure(figsize=(12, 8))
sns.pairplot(Auto, hue='mpg01', diag_kind='kde')
plt.show()

# Create boxplots to visualize the distribution of other features by 'mpg01'
plt.figure(figsize=(12, 8))
sns.boxplot(x='mpg01', y='cylinders', data=Auto)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='mpg01', y='horsepower', data=Auto)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='mpg01', y='weight', data=Auto)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='mpg01', y='acceleration', data=Auto)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='mpg01', y='origin', data=Auto)
plt.show()

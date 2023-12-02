import pandas as pd
import matplotlib.pyplot as plt

# Select the quantitative variables for which you want to create histograms
selected_variables = ['Apps', 'Enroll', 'Outstate', 'Expend']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Histograms of Selected Variables', fontsize=16)

# Create histograms for each selected variable with different numbers of bins
for i, variable in enumerate(selected_variables):
    row, col = divmod(i, 2)  # Calculate the subplot position (row, column)
    num_bins = 10 + i * 5  # Adjust the number of bins
    axes[row, col].hist(college[variable], bins=num_bins, edgecolor='black')
    axes[row, col].set_title(f'Histogram of {variable}')
    axes[row, col].set_xlabel(variable)
    axes[row, col].set_ylabel('Frequency')

# Adjust layout and display the subplots
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

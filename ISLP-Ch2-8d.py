import pandas as pd
import matplotlib.pyplot as plt

# Select the desired columns
columns_to_plot = ['Top10perc', 'Apps', 'Enroll']

# Create the scatterplot matrix
scatter_matrix = pd.plotting.scatter_matrix(college[columns_to_plot], figsize=(10, 10), diagonal='hist')

# Customize the plot labels and titles
plt.suptitle('Scatterplot Matrix of Top10perc, Apps, and Enroll', size=16)
for ax in scatter_matrix.ravel():
    ax.set_xlabel('')
    ax.set_ylabel('')

# Show the plot
plt.show()

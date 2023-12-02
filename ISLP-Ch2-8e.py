import pandas as pd
import matplotlib.pyplot as plt

# Create side-by-side boxplots
college.boxplot(column='Outstate', by='Private', grid=False)

# Customize the plot labels and titles
plt.title('Boxplot of Outstate by Private')
plt.suptitle('')  # Remove the default title

# Set axis labels
plt.xlabel('Private')
plt.ylabel('Outstate')

# Show the plot
plt.show()
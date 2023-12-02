import pandas as pd
import matplotlib.pyplot as plt

# Create the 'Elite' variable
college['Elite'] = pd.cut(college['Top10perc'], [0, 50, 100], labels=['No', 'Yes'])

# Check the number of elite universities
elite_counts = college['Elite'].value_counts()
print(elite_counts)

# Create side-by-side boxplots
college.boxplot(column='Outstate', by='Elite', grid=False)

# Customize the plot labels and titles
plt.title('Boxplot of Outstate by Elite')
plt.suptitle('')  # Remove the default title

# Set axis labels
plt.xlabel('Elite')
plt.ylabel('Outstate')

# Show the plot
plt.show()
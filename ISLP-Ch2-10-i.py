import pandas as pd
# Set the file path
file_path = r'D:\Phd-classes\Machine-learning\HM-1\Boston.csv'

# Read the data into a DataFrame
boston = pd.read_csv(file_path)

# Count suburbs with more than seven rooms per dwelling
more_than_seven_rooms = boston[boston['rm'] > 7]

# Count suburbs with more than eight rooms per dwelling
more_than_eight_rooms = boston[boston['rm'] > 8]

# Comment on suburbs with more than eight rooms
print("Suburbs with More than Seven Rooms per Dwelling:", more_than_seven_rooms.shape[0])
print("Suburbs with More than Eight Rooms per Dwelling:", more_than_eight_rooms.shape[0])

# Display information about suburbs with more than eight rooms
print("\nSuburbs with More than Eight Rooms per Dwelling:")
print(more_than_eight_rooms)
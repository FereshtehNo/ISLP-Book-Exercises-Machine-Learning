import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Define the file path
file_path = 'D:/Phd-classes/Machine-learning/HW3/Auto.csv'

# Read the CSV file into a DataFrame
Auto = pd.read_csv(file_path)

# Replace non-numeric values with NaN
Auto.replace('?', float('nan'), inplace=True)

# Drop rows with missing values
Auto.dropna(inplace=True)

# Convert 'horsepower' column to float (if applicable)
Auto['horsepower'] = Auto['horsepower'].astype(float)

# Define a threshold value (e.g., 25 mpg)
threshold = 25

# Create a binary categorical variable 'mpg01' based on the threshold
Auto['mpg01'] = (Auto['mpg'] > threshold).astype(int)

# Select the predictor variables
X = Auto[['cylinders', 'horsepower', 'weight', 'acceleration', 'origin'] ]

# Define the target variable
y = Auto['mpg01']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes model
naive_bayes = GaussianNB()

# Fit the model to the training data
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes.predict(X_test)

# Calculate the test error (accuracy)
test_accuracy = accuracy_score(y_test, y_pred)

# Print the test accuracy (test error)
print("Test Accuracy (Test Error): {:.2f}%".format(test_accuracy * 100))

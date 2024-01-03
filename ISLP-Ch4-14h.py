import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
X = Auto[['cylinders', 'horsepower', 'weight', 'acceleration', 'origin']]

# Define the target variable
y = Auto['mpg01']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of K values to try
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

best_accuracy = 0
best_k = None

# Loop over different values of K
for k in k_values:
    # Create a KNN model with the current K value
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model to the training data
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate the test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"K = {k}, Test Accuracy: {test_accuracy:.2f}")
    
    # Check if this K value has the best accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_k = k

print(f"The best K value is {best_k} with a test accuracy of {best_accuracy:.2f}")

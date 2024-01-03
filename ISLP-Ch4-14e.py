from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Define a threshold value (e.g., 25 mpg)
threshold = 25

# Create a new binary categorical variable 'mpg01' based on the threshold
Auto['mpg01'] = (Auto['mpg'] > threshold).astype(int)

# Split the dataset into training and testing sets
X = Auto[['cylinders', 'horsepower', 'weight', 'acceleration', 'origin']]
y = Auto['mpg01']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Quadratic Discriminant Analysis (QDA)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Make predictions on the test set
y_pred = qda.predict(X_test)

# Calculate the test error (accuracy)
test_accuracy = accuracy_score(y_test, y_pred)

# Print the test accuracy (test error)
print("Test Accuracy (Test Error): {:.2f}%".format(test_accuracy * 100))

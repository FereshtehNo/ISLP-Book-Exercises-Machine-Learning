import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load your dataset, replace '?' with NaN, and drop rows with missing values
Auto = pd.read_csv("D:/Phd-classes/Machine-learning/HW3/Auto.csv")
Auto = Auto.replace('?', np.nan)
Auto = Auto.dropna()

# Create 'mpg01' based on the threshold (e.g., 25)
Auto['mpg01'] = (Auto['mpg'] > 25).astype(int)

# Select features and target variable
X = Auto[['cylinders', 'horsepower', 'weight', 'acceleration', 'origin']]
y = Auto['mpg01']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit an LDA model to the training data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lda.predict(X_test)

# Calculate the test error using accuracy as a metric
test_error = 1 - accuracy_score(y_test, y_pred)

print("Test Error:", test_error)

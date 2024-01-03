from sklearn.model_selection import train_test_split

# Assuming 'Auto' is your DataFrame and 'mpg01' is the target variable

# Specify the features (X) and the target variable (y)
X = Auto.drop('mpg01', axis=1)  # Features, excluding 'mpg01'
y = Auto['mpg01']  # Target variable, 'mpg01'

# Split the data into a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the training and test sets to verify the split
print("Training set - X:", X_train.shape, "y:", y_train.shape)
print("Test set - X:", X_test.shape, "y:", y_test.shape)

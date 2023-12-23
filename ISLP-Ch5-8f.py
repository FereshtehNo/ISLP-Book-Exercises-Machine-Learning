import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# Set a random seed for reproducibility
np.random.seed(1)

# Generate the data and create a DataFrame
rng = np.random.default_rng(1)
X = rng.normal(size=100)
Y = X - 2 * X**2 + rng.normal(size=100)
data = pd.DataFrame({'X': X, 'Y': Y})

# Define the models for each case
models = {
    'Model 1': 1,
    'Model 2': 2,
    'Model 3': 3,
    'Model 4': 4
}

# Initialize variables to keep track of the best model and its corresponding error
best_model = None
best_mse = float('inf')  # Initialize with a large value

for model_name, degree in models.items():
    # Prepare the data for the model
    X_model = data[['X']].values
    Y_model = data['Y'].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_model)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_poly, Y_model)
    
    # Calculate LOOCV error
    loocv_scores = cross_val_score(model, X_poly, Y_model, cv=len(X_model), scoring='neg_mean_squared_error')
    
    # Calculate mean squared error
    mse = -loocv_scores.mean()
    
    # Check if this model has a lower error than the current best
    if mse < best_mse:
        best_mse = mse
        best_model = model_name

# Print the best model and its corresponding LOOCV error
print(f"The model with the smallest LOOCV error is {best_model} with a Mean Squared Error of {best_mse:.4f}")

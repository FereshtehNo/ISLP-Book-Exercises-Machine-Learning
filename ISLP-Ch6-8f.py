import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

# Set a random seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100

# Generate predictor X with a normal distribution (mean=0, std=1)
X = np.random.normal(0, 1, n)

# Generate noise vector epsilon with a normal distribution (mean=0, std=1)
epsilon = np.random.normal(0, 1, n)

# Chosen coefficients
beta0 = 2
beta7 = 3

# Generate the response vector Y based on the new model
Y = beta0 + beta7 * X**7 + epsilon

# Create a DataFrame to store the data
data = pd.DataFrame({'X7': X**7, 'Y': Y})

# Initialize an empty list to store selected predictors in forward stepwise selection
selected_predictors = []

# Initialize the best model based on Mallows' Cp for forward selection
best_model_forward = None
best_mallows_cp_forward = float('inf')

# Start forward stepwise selection
for _ in range(1):  # You want to select X7
    remaining_predictors = [col for col in data.columns if col not in selected_predictors]
    for predictor in remaining_predictors:
        model = sm.OLS(data['Y'], sm.add_constant(data[selected_predictors + [predictor]])).fit()
        current_mallows_cp = model.mse_total / model.scale
        if current_mallows_cp < best_mallows_cp_forward:
            best_mallows_cp_forward = current_mallows_cp
            best_model_forward = model
            best_predictor_forward = predictor
    selected_predictors.append(best_predictor_forward)

# Print the selected predictors and their coefficients for forward selection
print("Forward Stepwise Selection - Selected Predictors:")
print(selected_predictors)

print("Forward Stepwise Selection - Model Coefficients:")
print(best_model_forward.params)

# Define a range of alpha (λ) values to test for Lasso
alphas = np.logspace(-4, 4, 100)

# Perform Lasso cross-validation
lasso = LassoCV(alphas=alphas, cv=5)
lasso.fit(data[['X7']], data['Y'])

# Print the optimal alpha (λ) chosen by Lasso cross-validation
print("\nLasso - Optimal Alpha (λ):", lasso.alpha_)

# Fit the final Lasso model with the optimal lambda
lasso_final = LassoCV(alphas=[lasso.alpha_], cv=5)
lasso_final.fit(data[['X7']], data['Y'])

# Print the resulting coefficient estimates for Lasso
print("Lasso - Coefficient Estimates (Beta):")
print(lasso_final.coef_)

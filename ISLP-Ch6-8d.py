import statsmodels.api as sm

# Create a DataFrame to store the data
import pandas as pd
data = pd.DataFrame({'X': X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'X5': X**5, 'X6': X**6, 'X7': X**7, 'X8': X**8, 'X9': X**9, 'X10': X**10, 'Y': Y})

# Initialize the full model with all predictors
full_model = sm.OLS(data['Y'], sm.add_constant(data[['X', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']])).fit()
full_mallows_cp = full_model.mse_total / full_model.scale

# Initialize the best model based on Mallows' Cp
best_model = full_model
best_mallows_cp = full_mallows_cp

# Start backward stepwise selection
selected_predictors = list(data.columns[:-1])  # Start with all predictors

for _ in range(10):  # You want to remove up to X
    for predictor in selected_predictors:
        predictors_to_include = [col for col in selected_predictors if col != predictor]
        model = sm.OLS(data['Y'], sm.add_constant(data[predictors_to_include])).fit()
        current_mallows_cp = model.mse_total / model.scale
        if current_mallows_cp < best_mallows_cp:
            best_mallows_cp = current_mallows_cp
            best_model = model
            selected_predictors = predictors_to_include

# Print the selected predictors and their coefficients
print("Selected Predictors:")
print(selected_predictors)

print("Model Coefficients:")
print(best_model.params)

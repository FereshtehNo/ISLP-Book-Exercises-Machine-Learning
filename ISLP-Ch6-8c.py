import statsmodels.api as sm

# Create a DataFrame to store the data
import pandas as pd
data = pd.DataFrame({'X': X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'X5': X**5, 'X6': X**6, 'X7': X**7, 'X8': X**8, 'X9': X**9, 'X10': X**10, 'Y': Y})

# Initialize an empty list to store selected predictors
selected_predictors = []

# Initialize the best model based on Mallows' Cp
best_model = None
best_mallows_cp = float('inf')

# Start forward stepwise selection
for _ in range(10):  # You want to select up to X^10
    remaining_predictors = [col for col in data.columns if col not in selected_predictors]
    for predictor in remaining_predictors:
        model = sm.OLS(data['Y'], sm.add_constant(data[selected_predictors + [predictor]])).fit()
        current_mallows_cp = model.mse_total / model.scale
        if current_mallows_cp < best_mallows_cp:
            best_mallows_cp = current_mallows_cp
            best_model = model
            best_predictor = predictor
    selected_predictors.append(best_predictor)

# Print the selected predictors and their coefficients
print("Selected Predictors:")
print(selected_predictors)

print("Model Coefficients:")
print(best_model.params)

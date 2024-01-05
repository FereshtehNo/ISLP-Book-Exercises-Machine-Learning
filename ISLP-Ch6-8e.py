import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Create a DataFrame to store the data
data = pd.DataFrame({'X': X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'X5': X**5, 'X6': X**6, 'X7': X**7, 'X8': X**8, 'X9': X**9, 'X10': X**10, 'Y': Y})

# Define a range of alpha (λ) values to test
alphas = np.logspace(-4, 4, 100)

# Perform Lasso cross-validation
lasso = LassoCV(alphas=alphas, cv=5)
lasso.fit(data[['X', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']], data['Y'])

# Print the optimal alpha (λ) chosen by cross-validation
print("Optimal Alpha (λ):", lasso.alpha_)

# Create a plot of cross-validation error as a function of λ
mse_path = lasso.mse_path_
plt.figure(figsize=(10, 6))
plt.plot(np.log(lasso.alphas_), mse_path.mean(axis=1), marker='o', linestyle='-', color='b')
plt.xlabel('Log(Alpha / Lambda)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Error vs. Log(Alpha / Lambda)')
plt.grid(True)

# Fit the final Lasso model with the optimal lambda
lasso_final = LassoCV(alphas=[lasso.alpha_], cv=5)
lasso_final.fit(data[['X', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']], data['Y'])

# Print the resulting coefficient estimates
print("Coefficient Estimates (Beta):")
print(lasso_final.coef_)

# Show the plot
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Set the seed for reproducibility
np.random.seed(1)

# Generate a vector 'x' with 100 observations from N(0, 1) distribution
x = np.random.normal(0, 1, 100)

# Generate a vector 'eps' with 100 observations from N(0, 0.25) distribution
eps = np.random.normal(0, 0.25, 100)

# Calculate 'y' according to the specified linear model
y = -1 + 0.5 * x + eps

# Create and fit a linear regression model
model = LinearRegression()
x = x.reshape(-1, 1)  # Reshape 'x' to a 2D array for scikit-learn
model.fit(x, y)

# Calculate the residuals
residuals = y - model.predict(x)

# Calculate the standard error of the residuals
residual_std_error = np.std(residuals)

# Calculate the standard error of the coefficients (β0 and β1)
x_mean = np.mean(x)
x_std = np.std(x)
n = len(x)

# Calculate the standard errors
stderr_beta0 = residual_std_error * np.sqrt((1 / n) + (x_mean ** 2 / ((n - 1) * x_std ** 2)))
stderr_beta1 = residual_std_error / (x_std * np.sqrt(n - 1))

# Set the confidence level (e.g., 95% confidence interval)
confidence_level = 0.95

# Calculate the t-score for the desired confidence level
t_score = stats.t.ppf(1 - (1 - confidence_level) / 2, df=n - 2)

# Calculate the margin of error
margin_of_error_beta0 = t_score * stderr_beta0
margin_of_error_beta1 = t_score * stderr_beta1

# Calculate the confidence intervals
ci_beta0 = (model.intercept_ - margin_of_error_beta0, model.intercept_ + margin_of_error_beta0)
ci_beta1 = (model.coef_[0] - margin_of_error_beta1, model.coef_[0] + margin_of_error_beta1)

print("Confidence Interval for Beta0:", ci_beta0)
print("Confidence Interval for Beta1:", ci_beta1)

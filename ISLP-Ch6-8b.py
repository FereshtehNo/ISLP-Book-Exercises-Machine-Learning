# Chosen coefficients
beta0 = 2
beta1 = 3
beta2 = 1.5
beta3 = -0.5

# Generate the response vector Y using the model equation
Y = beta0 + beta1 * X + beta2 * X**2 + beta3 * X**3

# Print the first few values of Y for inspection
print("Response vector Y:")
print(Y[:10])  # Print the first 10 values of Y

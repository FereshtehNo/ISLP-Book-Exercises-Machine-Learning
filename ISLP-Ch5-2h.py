import numpy as np

# Set the parameters
n = 100  # Size of the bootstrap sample
j = 4    # The observation you want to check for

# Create an array to store the results
num_samples = 10000  # Number of bootstrap samples to generate
results = np.empty(num_samples)

# Original data (replace this with your dataset)
original_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Modify with your actual data

for i in range(num_samples):
    # Generate a bootstrap sample by resampling with replacement
    bootstrap_sample = np.random.choice(original_data, size=n, replace=True)
    
    # Check if the jth observation is in the bootstrap sample
    results[i] = j in bootstrap_sample

# Calculate the probability of j being in a bootstrap sample
probability = results.mean()

print(f"Probability of observation {j} being in a bootstrap sample: {probability:.4f}")

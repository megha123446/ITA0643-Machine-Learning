import numpy as np
from scipy.stats import norm

np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1, 100)
])

mu1, mu2 = np.random.rand(2) * 10
sigma1, sigma2 = np.random.rand(2) * 5 + 1e-6
pi = 0.5

max_iterations = 100
tolerance = 1e-6
prev_log_likelihood = 0

for _ in range(max_iterations):

    likelihood1 = norm.pdf(data, mu1, sigma1)
    likelihood2 = norm.pdf(data, mu2, sigma2)

    denominator = pi * likelihood1 + (1 - pi) * likelihood2 + 1e-10
    weight1 = (pi * likelihood1) / denominator
    weight2 = 1 - weight1

    mu1 = np.sum(weight1 * data) / np.sum(weight1)
    mu2 = np.sum(weight2 * data) / np.sum(weight2)

    sigma1 = np.sqrt(np.sum(weight1 * (data - mu1) ** 2) / np.sum(weight1))
    sigma2 = np.sqrt(np.sum(weight2 * (data - mu2) ** 2) / np.sum(weight2))

    pi = np.mean(weight1)

    log_likelihood = np.sum(np.log(denominator))

    if abs(log_likelihood - prev_log_likelihood) < tolerance:
        break

    prev_log_likelihood = log_likelihood

print("Final Parameters:")
print(f"Cluster 1 - Mean: {mu1:.2f}, Standard Deviation: {sigma1:.2f}")
print(f"Cluster 2 - Mean: {mu2:.2f}, Standard Deviation: {sigma2:.2f}")
print(f"Cluster Weights - Cluster 1: {pi:.2f}, Cluster 2: {1 - pi:.2f}")
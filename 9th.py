import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

plt.scatter(X, y, s=10, label='Data')
plt.plot(X, y_linear_pred, label='Linear Regression', color='red')
plt.plot(X, y_poly_pred, label='Polynomial Regression', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

mse_linear = mean_squared_error(y, y_linear_pred)
mse_poly = mean_squared_error(y, y_poly_pred)

print(f"Linear Regression MSE: {mse_linear:.4f}")
print(f"Polynomial Regression MSE: {mse_poly:.4f}")
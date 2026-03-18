import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

house_features = np.array([
    [1500, 3, 20],
    [2000, 4, 15],
    [1200, 2, 25]
])

house_prices = np.array([200000, 300000, 150000])

X_train, X_test, y_train, y_test = train_test_split(
    house_features, house_prices, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

new_house = np.array([[1800, 3, 18]])
print(f"Predicted Price: ${model.predict(new_house)[0]:,.2f}")
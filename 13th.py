import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

car_features = np.array([
    [2000, 150000, 4],
    [2010, 80000, 2],
    [2015, 50000, 1]
])

car_prices = np.array([5000, 15000, 25000])

X_train, X_test, y_train, y_test = train_test_split(
    car_features, car_prices, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

new_car = np.array([[2022, 10000, 2]])
print(f"Predicted Price: ${model.predict(new_car)[0]:,.2f}")
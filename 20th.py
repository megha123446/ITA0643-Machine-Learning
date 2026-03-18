from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

sales_features = np.array([[1], [2], [3], [4], [5]])
sales_targets = np.array([100, 150, 200, 250, 300])

X_train, X_test, y_train, y_test = train_test_split(
    sales_features, sales_targets, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Future Sales:")
for i, pred in enumerate(y_pred):
    print(f"Feature: {X_test[i][0]}, Predicted Sales: {pred:.2f}")
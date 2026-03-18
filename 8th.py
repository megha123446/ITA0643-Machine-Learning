from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_iris(return_X_y=True)

y_binary = (y == 0).astype(int)

np.random.seed(42)
y_noisy = y_binary + np.random.normal(scale=0.4, size=len(y_binary))

y_class = (y_noisy >= 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary) * 100
print(f"Accuracy: {accuracy:.2f}%")
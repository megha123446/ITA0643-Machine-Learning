from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

mobile_features = [
    [5.5, 32, 3, 13],
    [6.0, 64, 4, 15],
    [4.7, 16, 2, 10],
    [5.2, 32, 3, 12],
    [6.1, 128, 4, 16],
    [4.5, 16, 2, 9],
]

mobile_labels = np.array([1, 2, 1, 1, 3, 1])

np.random.seed(42)
mask = np.random.choice([True, False], size=len(mobile_labels), p=[0.3, 0.7])
mobile_labels[mask] = np.random.randint(1, 4, size=np.sum(mask))

X_train, X_test, y_train, y_test = train_test_split(
    mobile_features, mobile_labels, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

new_mobile = [[5.8, 64, 3, 14]]
print("Predicted Price Range:", model.predict(new_mobile)[0])
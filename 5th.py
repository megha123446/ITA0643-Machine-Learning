from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

np.random.seed(42)
noise = np.random.choice([0, 1], size=len(y), p=[0.9, 0.1])
y_noisy = (y + noise) % 3

X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.2, random_state=42
)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracy:.2f}%")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_iris(return_X_y=True)

np.random.seed(42)
mask = np.random.choice([True, False], size=len(y), p=[0.1, 0.9])
y_noisy = y.copy()
y_noisy[mask] = np.random.randint(0, 3, size=np.sum(mask))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
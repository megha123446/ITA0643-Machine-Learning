from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.2f}")

svm = SVC()
svm.fit(X_train, y_train)
print(f"SVM Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.2f}")

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(f"KNN Accuracy: {accuracy_score(y_test, knn.predict(X_test)):.2f}")
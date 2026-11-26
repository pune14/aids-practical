import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y_continuous = np.array([35,40,45,52,55,63,68,74,82,88])
y = (y_continuous >= 60).astype(int)
print("Class Labels:", y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
X_sorted = np.sort(X, axis=0)
y_prob = model.predict_proba(X_sorted)[:, 1]
plt.scatter(X, y, color="red", label="Actual Classes")
plt.plot(X_sorted, y_prob, color="blue", label="Logistic Regression Curve")
plt.axhline(0.5, color="gray", linestyle="--", label="Decision Threshold (0.5)")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([35,40,45,52,55,63,68,74,82,88])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope (w):", model.coef_[0])
print("Intercept (b):", model.intercept_)
y_pred = model.predict(X_test)
print("Test hours:", X_test.ravel())
print("Actual:", y_test)
print("Predicted:", y_pred.round(2))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), color="orange", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
    y_pred = model.predict(X_test)
    print(name)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Recall   :", recall_score(y_test, y_pred, average="macro"))
    print("F1-score :", f1_score(y_test, y_pred, average="macro"))
imp = pd.Series(rf.feature_importances_, index=data.feature_names)
imp.nlargest(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Features - Random Forest")
plt.grid(True)
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()

"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- pandas
- skcikit-learn
- matplotlib

The presented code utilizes a Support Vector Machine (SVM) to analyze the 'winequality-white.csv' dataset,
which comprises diverse chemical properties characterizing white wines, including acidity levels, residual sugar, and alcohol content.
Link to data model: https://machinelearningmastery.com/standard-machine-learning-datasets/
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Example: Using Random Forest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the new dataset with provided column names and skip the header row
column_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

df = pd.read_csv('winequality-white.csv', names=column_names, delimiter=';', skiprows=1)

# "quality" is the target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Example: Using Random Forest as an alternative model
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X_train_scaled, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test_scaled)

# Print model accuracy
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
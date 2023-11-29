"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- pandas
- skcikit-learn
- matplotlib

The provided code employs a Decision Tree Classifier to analyze the 'winequality-white.csv' dataset.
This dataset encompasses various chemical properties of white wines, such as acidity levels, residual sugar, and alcohol content.

Link to data model: https://machinelearningmastery.com/standard-machine-learning-datasets/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load the new dataset
column_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

df = pd.read_csv('winequality-white.csv', names=column_names, delimiter=';', skiprows=1)

# "quality" is the target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Decision Tree Classifier object
params = {'random_state': 0, 'max_depth': 8}
clf = DecisionTreeClassifier(**params)

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree
fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(clf,
                   feature_names=X.columns,
                   class_names=[str(c) for c in sorted(df['quality'].unique())],
                   filled=True)
plt.show()

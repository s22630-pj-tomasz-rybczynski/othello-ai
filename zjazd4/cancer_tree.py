"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- pandas
- skcikit-learn
- matplotlib

The problem this code solves is it trains the model to predict if the breast cancer is diagnosed or not using decision tree classification.
Link to data model: https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
df = pd.read_csv('Breast_cancer_data.csv')

# Split the dataset into features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Decision Tree Classifier object
params = {'random_state': 0, 'max_depth': 8}
clf = DecisionTreeClassifier(**params)

# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(clf, 
                   feature_names=X.columns,  
                   class_names=['0', '1'],
                   filled=True)

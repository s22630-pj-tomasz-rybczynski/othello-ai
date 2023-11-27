"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- pandas
- skcikit-learn
- matplotlib
- seaborn

The problem this code solves is it trains the model to predict if the breast cancer is diagnosed or not using C-Support vector classification.
Link to data model: https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Breast_cancer_data.csv')

# Split the dataset into features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a SVM Classifier object
clf = svm.SVC(kernel='linear')

# Train SVM Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

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

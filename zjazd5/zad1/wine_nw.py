"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- tensorflow
- matplotlib
- pandas
- sklearn
- os

Create model to guess wine quality
"""

import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load the dataset
data = pd.read_csv("winequality-white.csv", delimiter=";")

# Separate features (X) and target variable (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear"))  # Output layer for regression task

# Compile the model
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on Test Set: {mae}")

# Make predictions
predictions = model.predict(X_test)
print("Actual   Predicted")
for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(f"{actual}       {predicted[0]}")

plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
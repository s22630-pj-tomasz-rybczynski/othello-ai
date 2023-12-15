"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- tensorflow
- os
- numpy

Test dogs_check model
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('dog_breed_model_v2.h5')

# Path to the directory containing your image folders
data_dir = r'C:/Users/Filip/Desktop/Images'

# Get the names of subdirectories (assuming each subdirectory is a class)
class_labels = sorted([d.split('-')[-1] for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Define the path to your own image
img_path = 'awd.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Get the number of units in the last layer
num_units = model.layers[-1].output_shape[1]
print("Number of units in the last layer:", num_units)

# Ensure that num_units matches the length of class_labels
if len(class_labels) != num_units:
    raise ValueError("Number of units in the last layer does not match the length of class_labels")

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Get the class label from the predefined class labels
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print("Predicted Dog Breed:", predicted_class_label)

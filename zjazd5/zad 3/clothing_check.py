"""
Authors: Tomasz Rybczyński, Filip Marcoń

Precautions:
- Python 3.8
- tensorflow
- numpy

"""

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('my_model_20231207_220643.keras')


# Load and preprocess the test photo
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize image to match the model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to be between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (for grayscale)
    return img_array

# Specify the path to the test photo
test_photo_path = 'pullover.jpg'

# Preprocess the test photo
test_image = preprocess_image(test_photo_path)

# Make predictions
predictions = model.predict(test_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions[0])

# Map the class index to the corresponding class label
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

predicted_class_label = class_labels[predicted_class_index]

# Print the result
print(f'The predicted class is: {predicted_class_label}')
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt


def preprocess_image(image_path, target_size=(32, 32)):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Unable to load image from {image_path}")

    img = cv2.resize(img, target_size)  # Resize to the new target size
    img = img.astype('float32') / 255.0
    return img.reshape(1, *target_size, 3)


# Load the saved model

model = load_model('my_model_20231207_004700.keras')

# Load and preprocess a new image for prediction
image_path = 'deer.jpg'
img = preprocess_image(image_path)

# Make predictions
predictions = model.predict(img)
class_index = tf.argmax(predictions, axis=1).numpy()[0]

# Get the class label based on the CIFAR-10 dataset
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
predicted_class = class_labels[class_index]

# Convert image data type explicitly before displaying
img_display = (img.squeeze() * 255).astype('uint8')

# Visualize the input image and the predicted class
plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Class: {predicted_class}")
plt.show()

# Print the predicted class and softmax probability
softmax_prob = tf.reduce_max(tf.nn.softmax(predictions)).numpy()
print(f"The predicted class is: {predicted_class} with probability: {softmax_prob:.2%}")

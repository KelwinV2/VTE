import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import os

# Paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
model_path = os.path.join(parent_dir, 'models', 'testModel0-5-1.keras')
image_path = os.path.join(parent_dir, 'smaple', 'air_car.jpg')

# Load the trained model
resnet_model = load_model(model_path)

# Load and preprocess the image
img_height, img_width = 180, 180
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (img_height, img_width))
image_array = np.expand_dims(image_resized, axis=0)
image_array = preprocess_input(image_array)  # Preprocess the image as required by ResNet50

# Make a prediction
predictions = resnet_model.predict(image_array)

# Decode the predictions manually
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Update with your actual class names
predicted_indices = np.argsort(predictions[0])[::-1]  # Sort the predictions in descending order
top_3_indices = predicted_indices[:3]  # Get the top 3 predictions

# Print the top-3 predicted classes
for i in top_3_indices:
    if predictions[0][i] > 0.1:
        print(f"{class_names[i]}: {predictions[0][i] * 100:.1f}%")

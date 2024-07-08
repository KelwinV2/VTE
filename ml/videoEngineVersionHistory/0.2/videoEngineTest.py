import tensorflow as tf
from PIL import Image
import numpy as np
from data.classes import classes

# Load the saved model
model = tf.keras.models.load_model("models/testModel0-2.keras")

# Function to preprocess the image
def preprocess_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)
    # Resize the image to match the input shape of the model (32x32)
    img = img.resize((32, 32))
    # Convert the image to a numpy array and normalize the pixel values
    img_array = np.array(img) / 255.0
    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Path to the image
image_path = "sample/air_car.jpg"

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(preprocessed_image)

# Apply a threshold to determine which classes are present
threshold = 0.5
predicted_labels = (predictions[0] > threshold).astype(int)

# Get the predicted class labels
predicted_classes = [classes[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]

print("Predicted classes:", predicted_classes)

import tensorflow as tf
from PIL import Image
import numpy as np
from data.classes import classes

# Load the saved model
model = tf.keras.models.load_model("models/testModel.keras")

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

# Path to the PNG image
image_path = "sample/cat.jpg"

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(preprocessed_image)

# Get the predicted class label
predicted_class_index = np.argmax(predictions)
predicted_class = classes[predicted_class_index]

print("Predicted class:", predicted_class)
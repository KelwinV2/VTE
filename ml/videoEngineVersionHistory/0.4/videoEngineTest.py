import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

# Load the saved model
model = tf.keras.models.load_model("models/testModel0-4.keras")

# Paths
csv_file_path = 'data/multilabel_classification(6)-reduced_modified.csv'

# Load the CSV file to get the class names
df = pd.read_csv(csv_file_path)
classes = df.columns[2:].tolist()  # Assuming the first two columns are 'Image_Name' and the list of classes

# Function to preprocess the image
def preprocess_image(image_path):
    try:
        # Open the image using PIL
        img = Image.open(image_path)
        # Resize the image to match the input shape of the model (32x32)
        img = img.resize((32, 32))
        # Convert the image to a numpy array and normalize the pixel values
        img_array = np.array(img) / 255.0
        # Expand the dimensions to match the input shape expected by the model
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Path to the image
image_path = "sample/motor_boat.jpg"

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

if preprocessed_image is not None:
    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Apply a threshold to determine which classes are present
    threshold = 0.5
    predicted_labels = (predictions[0] > threshold).astype(int)

    # Get the predicted class labels
    predicted_classes = [classes[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]

    print(classes)

    print("Raw predictions:", predictions)
    print("Predicted classes:", predicted_classes)
else:
    print("Failed to preprocess the image.")

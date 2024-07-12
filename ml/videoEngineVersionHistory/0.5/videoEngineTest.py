import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# specifying model path
import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
# Construct the absolute path to the data file
model_file_path = os.path.join(parent_dir, 'models', 'testModel0-5.keras')
image_path = os.path.join(parent_dir, 'sample', 'fish.jpeg')

# Load the saved model
model = tf.keras.models.load_model(model_file_path)

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Decode and print the top-3 predicted classes
decoded_predictions = decode_predictions(predictions, top=5)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    if score > 0.1:
        print(label, "{:.1f}%".format(score * 100))

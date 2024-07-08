import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load pre-trained ResNet50 model trained on imagenet dataset
model = ResNet50(weights='imagenet', include_top=True)

# Summary of the model architecture
model.summary()

# Save the model
model.save("./models/testModel0-5.keras")

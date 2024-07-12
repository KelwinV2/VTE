import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pathlib

# Paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# Load dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Print data directory
print(data_dir)

# Define image size and batch size
img_height, img_width = 180, 180
batch_size = 32

# Load training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load validation data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names

# Plot some sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Create ResNet50 base model
pretrained_model = ResNet50(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg', weights='imagenet')

# Freeze the base model
for layer in pretrained_model.layers:
    layer.trainable = False

# Add new layers on top of the ResNet50 base model
inputs = keras.Input(shape=(img_height, img_width, 3))
x = pretrained_model(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
resnet_model = Model(inputs, outputs)

# Compile the model
resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Save the model in Keras format
# Construct the absolute path to the class data file
model_save_path = os.path.join(parent_dir, 'models', 'testModel0-5-1.keras')
resnet_model.save(model_save_path)

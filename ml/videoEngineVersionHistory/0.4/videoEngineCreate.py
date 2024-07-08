import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Paths
image_folder_path = 'data/images'
csv_file_path = 'data/multilabel_classification(6)-reduced_modified.csv'

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Extract image paths
image_paths = df['Image_Name'].apply(lambda x: os.path.join(image_folder_path, x)).values

# Extract binary labels
binary_labels = df.iloc[:, 2:].values  # Assuming the first column is image names and the rest are binary labels

# Load images and resize to 32x32
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(32, 32))
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image
    return image

X = np.array([load_and_preprocess_image(img_path) for img_path in image_paths])
y = binary_labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and Compiling our convolutional neural network
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_train.shape[1], activation='sigmoid')  # Change to sigmoid for multi-label
])

cnn.compile(optimizer='adam',
            loss='binary_crossentropy',  # Change to binary cross-entropy
            metrics=['accuracy'])

# Training our CNN
history = cnn.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Evaluating our CNN
cnn.evaluate(X_test, y_test)

# Saving the Model
cnn.save("./models/testModel0-4.keras")

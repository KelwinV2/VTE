import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from tensorflow.keras.utils import to_categorical
from data.classes import classes

# Using a provided dataset from keras that contains a large series of images
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalizing the training data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert the labels to a multi-label format
y_train_multilabel = to_categorical(y_train, num_classes=10)
y_test_multilabel = to_categorical(y_test, num_classes=10)

# Building and Compiling our convolutional neural network
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='sigmoid')  # Change to sigmoid for multi-label
])

cnn.compile(optimizer='adam',
            loss='binary_crossentropy',  # Change to binary cross-entropy
            metrics=['accuracy'])

# Training our CNN
history = cnn.fit(X_train, y_train_multilabel, epochs=20, validation_data=(X_test, y_test_multilabel))

# Evaluating our CNN
cnn.evaluate(X_test, y_test_multilabel)

# Saving the Model
cnn.save("./models/testModel0-2.keras")

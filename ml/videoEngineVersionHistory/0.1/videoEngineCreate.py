import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from data.classes import classes

# Using a provided dataset from keras that contains a large series of images
(X_train, y_train) , (X_test, y_test) = datasets.cifar10.load_data()

# Normalizing the training data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Building and Compiling our convolutional neural network
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Training our CNN
cnn.fit(X_train, y_train, epochs=10)

# Evaluating our CNN
cnn.evaluate(X_test,y_test)

# Saving the Model
cnn.save("./models/testModel.keras")







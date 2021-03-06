# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ImzOxXbTHnRrddm144_oXxdmX33Nk8VJ
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

def create_LeNet():

    model = keras.Sequential([
        keras.layers.Conv2D(128, kernel_size=4, padding='same', input_shape=(28, 28,1), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2,strides=2),
        keras.layers.Conv2D(128, kernel_size=4, padding='same', input_shape=(28, 28,1), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2,strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":

    number_epochs = 30

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    model = create_LeNet()

    img_process = ImageDataGenerator(horizontal_flip=1)
    history = model.fit_generator(img_process.flow(train_images,train_labels,batch_size=100), steps_per_epoch=len(train_images)/100, epochs=number_epochs)
    plt.plot(history.history['loss'],label="Training Loss")
    plt.title('Training Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    model.save('./convnet_model.h5')

loss, acc = model.evaluate(test_images, test_labels)
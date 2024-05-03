import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, ReLU

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
print(tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:1'):
# Loading and preprocessing the dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.

    def model():
        model = Sequential()
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
        model.add(ReLU())
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
        model.add(ReLU())
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(ReLU())
        model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
        model.add(ReLU())
        model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'))
        model.add(ReLU())
        model.add(Conv2DTranspose(3, (3, 3), padding='same'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        return model

    aeModel = model()
    aeModel.fit(train_images, train_images, epochs =20, batch_size=32, validation_data=(test_images, test_images), verbose=1)
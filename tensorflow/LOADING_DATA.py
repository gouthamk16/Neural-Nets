

############## FOR LOADING IMAGE FROM FOLDER ################


import torch
import cv2
import zipfile
import torchvision
from torchvision import datasets, transforms
import os
from torch import nn
import torch.nn.functional as F

device = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'

print(f"Device : {device}")

transform = transforms.Compose([
    transforms.Resize(size=(227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
])

train_path = 'C:/Users/Goutham/Downloads/pytorch/dogs_vs_cats/train'
test_path = 'C:/Users/Goutham/Downloads/pytorch/dogs_vs_cats/test'


train_data = datasets.ImageFolder(train_path, transform=transform)
test_data = datasets.ImageFolder(test_path, transform=transform)

trainloader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = False
)
testloader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = 32,
    shuffle = False
)

train_features_batch, train_labels_batch = next(iter(trainloader))

print(train_data.class_to_idx)



################# loading data from dataset ###################
import torch
import torch.nn as nn
import torchvision

# Loading and processing the dataset
trainset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    download = False,
    transform = torchvision.transforms.ToTensor(),
    target_transform = None
)
testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = False,
    transform = torchvision.transforms.ToTensor(),
    target_transform = None
)

trainloader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = 32,
    shuffle = True
)
testloader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size = 32,
    shuffle = True
)

############### FOR TEnsorflow #############

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


########### FROM FOLDER ###################

import tensorflow as tf
from keras import layers, Model
from keras.preprocessing.image import ImageDataGenerator

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/Goutham/Downloads/pytorch/dogs_vs_cats/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/Goutham/Downloads/pytorch/dogs_vs_cats/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Define the model
def cats_model():
    inputs = layers.Input(shape=(64, 64, 3))
    x = layers.Conv2D(96, (3, 3), strides=4, activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    x = layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    # x = layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D((3, 3), strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = cats_model()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Save the model
model.save('catsvdogsModel_tf_gpu.h5')

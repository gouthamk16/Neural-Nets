

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
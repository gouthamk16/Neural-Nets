# Implementing the VGG16 using pytorch
# TO DO:
# Resnet50, resnet152, Mobilenet, alexnet, convnext

import torch
import torch.nn as nn
from torch.nn import functional
import torchvision
from torchvision import transforms
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
])

trainset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    download = False,
    transform = transform,
    target_transform = None
)
testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = False,
    transform = transform,
    target_transform = None
)

trainloader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = 40,
    shuffle = True
)
testloader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size = 40,
    shuffle = True
)

# Creating the VGG16 model
def vgg16Model(num_features):
    model = models.vgg16(pretrained = True)
    input_lastlayer = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(input_lastlayer, num_features)
    model = model.to(device)
    return model

# Creating the ResNet50
def resnet50Model(num_features):
    model = models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True
    input_lastlayer = model.fc.in_features
    model.fc = nn.Linear(input_lastlayer, num_features)
    model = model.to(device)
    return model

# def vgg16Model(num_features):
#     model = models.vgg16(pretrained = True)
#     input_lastlayer = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(input_lastlayer, num_features)
#     model = model.to(device)
#     return model

# def vgg16Model(num_features):
#     model = models.vgg16(pretrained = True)
#     input_lastlayer = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(input_lastlayer, num_features)
#     model = model.to(device)
#     return model

# def vgg16Model(num_features):
#     model = models.vgg16(pretrained = True)
#     input_lastlayer = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(input_lastlayer, num_features)
#     model = model.to(device)
#     return model


model = resnet50Model(num_features=10)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
opt = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=5e-4)


# Creating the training loops
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for i, (image, labels) in enumerate(trainloader):
        image, labels = image.to(device), labels.to(device)
        opt.zero_grad()
        y_pred = model(image)
        # loss = F.cross_entropy(y_pred, labels)
        # loss = F.nll_loss(y_pred, labels)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        opt.step()
        # Print the loss every 2000 mini batches
        running_loss += loss
        if i%100==99:
            print(f"Epoch {epoch+1}/{epochs}({i+1}/{len(trainloader)}) | Training Loss: {running_loss/100}")
            running_loss = 0.0 # Reset the running loss


    # Validation loop
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (test_image, test_labels) in enumerate(testloader):
            test_image, test_labels = test_image.to(device), test_labels.to(device)
            y_test = model(test_image)
            test_loss += loss_fn(y_test, test_labels).item()
            _, predicted = torch.max(y_test.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    test_loss = test_loss / len(testloader)
    test_acc = correct / total

    print(f"Validation loss after epoch {epoch+1}: {test_loss} | Validation Accuracy: {test_acc*100}%")


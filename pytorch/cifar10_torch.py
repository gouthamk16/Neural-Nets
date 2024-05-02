## Performing image classification on the CIFAR-10 dataset

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# Loading and processing the dataset

transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # Mean and standard deviation used in the proessing of image net
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
    batch_size = 32,
    shuffle = True
)
testloader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size = 32,
    shuffle = True
)

# Creating the Convolution Model
class cifar10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2, 2), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=1, padding=1)
        self.act3 = nn.ReLU()
        #self.batch1 = nn.BatchNorm2d()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(1296, 512)
        self.act4 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):

        # input 3x32x32 | output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32 | output 32x33x33
        x = self.act2(self.conv2(x))
        # input 32x33x33 | output 32x32x32
        x = self.pool2(x)
        # input 32x32x32 | output 16x34x34
        x = self.act3(self.conv3(x))
        #x = self.batch1(x)
        # input 16x34x34 | output 16x33x33
        x = self.pool3(x)
        # input 16x33x33 | output 17424
        x = self.flat(x)
        # input 17424 | output 512
        x = self.act4(self.fc3(x))
        x = self.drop3(x)
        # input 512 | outpupt 10
        x = self.fc4(x)
        

        return x

model = cifar10Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr=0.01, momentum=0.9)

# Training loop
epochs = 20

for epoch in range(epochs):
    model.train()
    for inputs, labels in trainloader:
        # forward, backward and weight update
        
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0

    model.eval()
    for inputs, labels in testloader:
        with torch.inference_mode():
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1)==labels).float().sum()
            count += len(labels)
    acc = acc / count
    print(f"Epoch: {epoch} | Accuracy: {acc*100}")


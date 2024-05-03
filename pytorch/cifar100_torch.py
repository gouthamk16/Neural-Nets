## Performing image classification on the CIFAR-10 dataset

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Loading and processing the dataset

transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # Mean and standard deviation used in the proessing of image net
    )
])

trainset = torchvision.datasets.CIFAR100(
    root = "./data",
    train = True,
    download = True,
    transform = transform,
    target_transform = None
)
testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
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

        self.fc4 = nn.Linear(512, 100)

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
opt = torch.optim.SGD(params = model.parameters(), lr=0.01, momentum=0.9)

# Training loop
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

    with torch.inference_mode():
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

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
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
])

train_path = 'C:/Users/Goutham/Downloads/pytorch/data/dogs_vs_cats/train'
test_path = 'C:/Users/Goutham/Downloads/pytorch/data/dogs_vs_cats/test'


train_data = datasets.ImageFolder(train_path, transform=transform)
test_data = datasets.ImageFolder(test_path, transform=transform)

trainloader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = True
)
testloader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = 32,
    shuffle = True
)

train_features_batch, train_labels_batch = next(iter(trainloader))

print(train_data.class_to_idx)

class catsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=(3, 3), padding=0, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(2700, 512)
        self.drop1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(512, 64)
        self.dense3 = nn.Linear(64, 1)
    def forward (self, x):
        x = self.pool1(self.conv1(x))
        x = F.relu(x)
        x = self.flat(x)
        x = F.relu(self.dense1(x))
        x = self.drop1(x)
        x = F.relu(self.dense2(x))
        # x = F.log_softmax(self.dense3(x))
        x = self.dense3(x)
        return x
    

# Creating the training and validation loops
model = catsModel().to(device)
opt = torch.optim.Adam(params = model.parameters(), lr = 0.1)
loss_fn = nn.BCEWithLogitsLoss() # Also computes the log sigmoid of the output
epochs = 20

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for i, (image, labels) in enumerate(trainloader):
        labels = labels.unsqueeze(1).float()
        # print(labels.shape)
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
            test_labels = test_labels.unsqueeze(1).float()
            test_image, test_labels = test_image.to(device), test_labels.to(device)
            y_test = model(test_image)
            test_loss += loss_fn(y_test, test_labels).item()
            predicted = (y_test > 0.5).float()  # Applying threshold 0.5 for binary classification
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    test_loss = test_loss / len(testloader)
    test_acc = correct / total

    print(f"Validation loss after epoch {epoch+1}: {test_loss} | Validation Accuracy: {test_acc*100}%")


torch.save(model.state_dict(), "catsvdogsModel.pth")
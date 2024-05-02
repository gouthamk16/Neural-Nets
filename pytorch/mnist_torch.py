import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5), (0.5)) # Mean = 0.5, SD = 0.5
])

trainset = MNIST(
    root = './data', 
    train = True,
    download = False, 
    transform = transform,
    target_transform = None
)
validationset = MNIST(
    root = './data', 
    train = False,
    download = False, 
    transform = transform,
    target_transform = None
)

trainloader = DataLoader(
    dataset = trainset,
    batch_size = 32,
    shuffle = True
)

testloader = DataLoader(
    dataset = validationset,
    batch_size = 32,
    shuffle = True
)

sample_batch_images, sample_batch_labels = next(iter(trainloader))

print("Classes: ", trainset.class_to_idx)
print("Sample batch shape: ", sample_batch_images.shape)
print("Sample labels batch shape: ", sample_batch_labels.shape)

# for i, data in enumerate(trainloader):
#     print(i)


# If you are working on a multi-class classification use case and use nn.CrossEntropyLoss, 
# your model should output raw logits, as internally nn.CrossEntropyLoss will apply 
# F.log_softmax and nn.NLLLoss.

# Use the nll_loss for multiclass classification problem along with softmax ( if you want to use it indipendently)


# # Create the model
class mnist_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=(3, 3), padding=0, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(2700, 512)
        self.drop1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(512, 64)
        self.dense3 = nn.Linear(64, 10)
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

# Epochs, loss function, model and optimizer
# Use the nll_loss for multiclass classification problem along with softmax
model = mnist_classifier().to(device)
epochs = 5
opt = torch.optim.SGD(params = model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

# for epoch in range(epochs):
#     running_loss = 0.0
#     model.train()
#     for i, (image, labels) in enumerate(trainloader):
#         image, labels = image.to(device), labels.to(device)
#         opt.zero_grad()
#         y_pred = model(image)
#         # loss = F.cross_entropy(y_pred, labels)
#         # loss = F.nll_loss(y_pred, labels)
#         loss = loss_fn(y_pred, labels)
#         loss.backward()
#         opt.step()
#         # Print the loss every 2000 mini batches
#         running_loss += loss
#         if i%100==99:
#             print(f"Epoch {epoch+1}/{epochs}({i+1}/{len(trainloader)}) | Training Loss: {running_loss/100}")
#             running_loss = 0.0 # Reset the running loss


#     # Validation loop
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for i, (test_image, test_labels) in enumerate(testloader):
#             test_image, test_labels = test_image.to(device), test_labels.to(device)
#             y_test = model(test_image)
#             test_loss += loss_fn(y_test, test_labels).item()
#             _, predicted = torch.max(y_test.data, 1)
#             total += test_labels.size(0)
#             correct += (predicted == test_labels).sum().item()

#     test_loss = test_loss / len(testloader)
#     test_acc = correct / total

#     print(f"Validation loss after epoch {epoch+1}: {test_loss} | Validation Accuracy: {test_acc*100}%")

# Save the model
# torch.save(model.state_dict(), "mnist_model.pth")

# Load the model
model.load_state_dict(torch.load('mnist_model.pth'))

# Predicting a sample image
for test_image, test_label in testloader:
    sample_image = test_image[0]
    sample_label = test_label[0]

    # View the image
    img_array = sample_image.squeeze().numpy()
    print(img_array.shape)
    plt.imshow(img_array)
    plt.show()

    sample_image = sample_image.unsqueeze(1)

    model.eval()
    with torch.inference_mode():
        prob_prediction = model(sample_image)
        _, prediction = torch.max(prob_prediction, 1)
        print(prediction)

    break
    # sample_image, sample_label = sample_image.to(device), sample_label.to(device)

# CReating an autoencoder for thr cifar10 dataset

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# # Creating the Convolution Model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(7, 7))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(params = model.parameters(), lr=0.01, momentum=0.9)

# Training loop
epochs = 30

# for epoch in range(epochs):
#     running_loss = 0.0
#     model.train()
#     for i, (image, _) in enumerate(trainloader):
#         image = image.to(device)
#         opt.zero_grad()
#         y_pred = model(image)
#         # loss = F.cross_entropy(y_pred, labels)
#         # loss = F.nll_loss(y_pred, labels)
#         loss = loss_fn(y_pred, image)
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

#     with torch.no_grad():
#         for i, (test_image, _) in enumerate(testloader):
#             test_image = test_image.to(device)
#             y_test = model(test_image)
#             test_loss += loss_fn(y_test, test_image).item()
#             _, predicted = torch.max(y_test.data, 1)

#     test_loss = test_loss / len(testloader)

#     print(f"Validation loss after epoch {epoch+1}: {test_loss}")

# # Save the model
# torch.save(model.state_dict(), "cifar_autoencoder.pth")

# # Load the model
model.load_state_dict(torch.load('cifar_autoencoder.pth'))


# Testing the predictions/generations of the autoencoder
for test_image, _ in testloader:
    sample_image = test_image[0]

    fig, ax = plt.subplots(1, 2)

    # View the image
    img_array = sample_image.squeeze().numpy()
    reshaped_image = np.transpose(img_array, (1, 2, 0))
    print(reshaped_image.shape)
    ax[0].imshow(reshaped_image)

    # sample_image = sample_image.unsqueeze(1)

    model.eval()
    with torch.inference_mode():
        prob_prediction = model(sample_image)
        img_array_2 = prob_prediction.squeeze().numpy()
        reshaped_image_2 = np.transpose(img_array_2, (1, 2, 0))
        ax[1].imshow(reshaped_image_2)
    
    plt.show()

    break
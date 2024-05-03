import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot


class convents(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=4, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.dropout1(x)
		x = self.pool(F.relu(self.conv2(x)))
		x = self.dropout2(x)
		x = x.view(-1, 64 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
	
net = convents()

# Loss functions and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net


img = Image.open("Ganesh.jpg")
img

# transform the image to pytorch tensor
transform = transforms.Compose([
	transforms.ToTensor(),
])


img = transform(img)
img.shape


# Get activations of the first convolutional layer
conv1 = net.conv1
print('First convolution Layer :',conv1)

# APPLY THE FIRST Convolutional layer to the image
y = conv1(img)
print('Output Shape :',y.shape)

make_dot(y.mean(), params=dict(conv1.named_parameters()))

# Get weights of the first convolutional layer
weights = conv1.weight.detach().numpy()
weights.shape

plt.figure(figsize =(12,5))

# Plot the original grayscale image
plt.subplot(1,3,1)
plt.imshow(img[0],cmap = 'gray')
plt.title('Original')
plt.axis('off')
img = Variable(img.unsqueeze(0), requires_grad=True)


#Plot the convolved grayscale image

# Squeeze tensor to numpy image
img_conv1 = y.detach().numpy()
img_conv1 = np.squeeze(img_conv1)

plt.subplot(1,3,2)
plt.imshow(img_conv1[0], cmap = 'gray')
plt.axis('off')
plt.title('After convolutions')

# Plot the weights of the first convolutional layer

# Squeeze tensor to numpy image
weights = np.squeeze(weights)

plt.subplot(1,3,3)
plt.imshow(weights[0,0,:,:], cmap='gray')
plt.axis('off')
plt.title('First convolutions weights')
plt.show()


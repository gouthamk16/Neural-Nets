import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import imageio
from torchvision.transforms.functional import to_pil_image
from torch.autograd.variable import Variable
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the cifar10 dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

trainset = MNIST(root='./data', download = False, train = True, transform = transform)
testset = MNIST(root='./data', download = False, train = False, transform = transform)

trainloader = DataLoader(dataset = trainset, batch_size = 32, shuffle = True)
testloader = DataLoader( dataset = testset, batch_size = 32, shuffle = True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 128
        self.out_features = 784
        self.fc0 = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.out_features),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc0(x)
        # print(x.shape)
        x = x.view(-1, 1, 28, 28)
        # print(x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 784
        self.out_features = 1
        self.fc0 = nn.Sequential(
            nn.Linear(self.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.out_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc0(x)
        return x
    

generator = Generator()
discrim = Discriminator()

generator.to(device)
discrim.to(device)

g_opt = torch.optim.Adam(params=generator.parameters(), lr=2e-4)
d_opt = torch.optim.Adam(params=discrim.parameters(), lr=2e-4)

g_losses = []
d_losses = []
gen_images = []

loss_fn = nn.BCELoss()

def noise(n, n_features = 128):
    return torch.randn(n, n_features).to(device)

def make_ones(size):
    return torch.ones(size, 1).to(device)

def make_zeros(size):
    return torch.zeros(size, 1).to(device)

def train_discrim(opt, real_data, fake_data):
    n = real_data.size(0)
    opt.zero_grad()
    prediction_real = discrim(real_data)
    error_real = loss_fn(prediction_real, make_ones(n))
    error_real.backward()

    prediction_fake = discrim(fake_data)
    error_fake = loss_fn(prediction_fake, make_zeros(n))

    error_fake.backward()
    opt.step()

    return error_real + error_fake

def train_generator(opt, fake_data):
    n = fake_data.size(0)
    opt.zero_grad()
    prediction = discrim(fake_data)
    error = loss_fn(prediction, make_ones(n))
    error.backward()
    opt.step()
    return error

num_epochs = 25
k = 1
test_noise = noise(32)

for epoch in range(num_epochs):
    g_error = 0
    d_error = 0
    generator.train()
    discrim.train()
    for i, (images, _) in enumerate(trainloader):
        images = images.to(device)
        n = len(images)
        for j in range(k):
            fake_data = generator(noise(n))
            real_data = images.to(device)
            d_error += train_discrim(d_opt, real_data, fake_data)
        fake_data = generator(noise(n))
        g_error += train_generator(g_opt, fake_data)
        if i%100==99:
            print(f"Epoch: {epoch}/{num_epochs}({i}/{len(trainloader)}) | d_loss: {d_error/100} | g_loss: {g_error/100}")
            d_error = 0
            g_error = 0
    img = generator(test_noise).cpu().detach()
    img = make_grid(img)
    gen_images.append(img)
    g_losses.append(g_error/i)
    d_losses.append(d_error/i)
    # print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, g_error/i, d_error/i))


print('Training Finished')
torch.save(generator.state_dict(), 'mnist_generator.pth')

def to_image(tensor):
    """Converts a PyTorch tensor to a PIL Image."""
    return to_pil_image(tensor)

import numpy as np
from matplotlib import pyplot as plt
imgs = [np.array(to_image(i)) for i in gen_images]
imageio.mimsave('progress.gif', imgs)
    

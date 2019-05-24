import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import torch.utils as utils
from torchvision.utils import save_image

hidden_dim = 64
batch_size = 256

transform=transforms.Compose([transforms.ToTensor()])
train_db = datasets.MNIST("./", train=True, download=True,
    transform = transform)
test_db = datasets.MNIST("./", train=False, download=True,
    transform=transform)

train_loader = utils.data.DataLoader(train_db, batch_size=batch_size, shuffle = True)
test_loader = utils.data.DataLoader(test_db, shuffle = True)

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, 300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, 100),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

        self.creat_mu = nn.Linear(100, hidden_dim)

        self.creat_sigma = nn.Linear(100, hidden_dim)

    def reparameterize(self, mu, sigma):
        eps = torch.randn(mu.shape[0], hidden_dim)
        z = mu + eps * torch.exp(sigma / 2)
        return z

    def forward(self, x):
        to_creat_mu, to_creat_sigma = self.encoder(x), self.encoder(x)
        mu = self.creat_mu(to_creat_mu)
        sigma = self.creat_sigma(to_creat_sigma)
        z = self.reparameterize(mu, sigma)
        out = self.decoder(z)
        return out, mu, sigma

def loss_func(out, x):
    BCE = F.mse_loss(out, x)
    return BCE


vae = VAE()


optimizer = optim.Adam(vae.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

epochs = 1000


for epoch in range(epochs):
    total_loss = 0.
    for index, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.view(data.shape[0], -1)
        out, mu, sigma = vae(data)
        loss = loss_func(out, data)
        print(mu)
        print(sigma)
        loss.backward()
        total_loss += loss
        optimizer.step()





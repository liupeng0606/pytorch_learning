import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import torch.utils as utils
from torchvision.utils import save_image

hidden_dim = 64
batch_size = 16

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
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc11 = nn.Linear(128 * 7 * 7, hidden_dim)
        self.fc12 = nn.Linear(128 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128 * 7 * 7)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )



    def reparameterize(self, mu, sigma):
        eps = torch.randn(mu.shape[0], hidden_dim)
        z = mu + eps * torch.exp(sigma / 2)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 128, 7, 7)  # batch_s, 8, 7, 7

        return self.decoder(out3), mu, logvar

def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


vae = VAE()


optimizer = optim.Adam(vae.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

epochs = 1000


for epoch in range(epochs):
    total_loss = 0.
    for index, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_x, mu, logvar = vae.forward(data)
        loss = loss_func(recon_x, data, mu, logvar)
        loss.backward()
        total_loss += loss
        optimizer.step()


        if index % 50 == 0:
            sample = torch.randn(64, hidden_dim)
            sample = vae.decoder(vae.fc2(sample).view(64, 128, 7, 7))
            save_image(sample.data.view(64, 1, 28, 28),
                       './img/sample_' + str(epoch) + '.png')
            print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                epoch, index * len(data), len(train_loader.dataset),
                       100. * index / len(train_loader), loss / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(train_loader.dataset)))




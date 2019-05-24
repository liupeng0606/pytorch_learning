import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import torch.utils as utils
from torchvision.utils import save_image

hidden_dim = 64
batch_size = 100

# transform=transforms.Compose([transforms.ToTensor()])
# train_db = datasets.MNIST("./", train=True, download=True,
#     transform = transform)
# test_db = datasets.MNIST("./", train=False, download=True,
#     transform=transform)
#
# train_loader = utils.data.DataLoader(train_db, batch_size=batch_size, shuffle = True)
# test_loader = utils.data.DataLoader(test_db, shuffle = True)

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
        )


    def sample_data(self, mu):
        eps = torch.randn(3, 5)
        xx = mu @ eps
        return xx




    def forward(self, x):

        to_creat_sigma = self.encoder(x)
        z = self.sample_data(to_creat_sigma)
        out = self.decoder(z)
        return out



vae = VAE()

vv = list(vae.named_parameters())

print(vv)

print(len(vv))


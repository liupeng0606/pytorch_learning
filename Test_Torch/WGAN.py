import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random


h_dim = 400
batch_size = 256
vis = visdom.Visdom()

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )
    def forward(self, z):
        return self.net(z)


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z).view(-1)


def data_generator():
    scale = 2.
    center = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x , scale * y) for x, y in center]
    while True:
        data_set = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            data_set.append(point)
        data_set = np.array(data_set).astype(np.float32)
        data_set /= 1.414
        yield data_set

def gradient_penalty(d, x, fake_data):

    t =  torch.rand(batch_size, 1)
    t = t.expand_as(x)
    mid = t * x + (1 - t) * fake_data
    mid.requires_grad=True
    mid.requires_grad_()
    out = d(mid)
    grads = autograd.grad(outputs=out, inputs=mid,
                          grad_outputs=torch.ones_like(out),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


def main():
    torch.manual_seed(15)
    np.random.seed(15)
    data_iter = data_generator()
    # x = next(data_iter)
    g = G()
    d = D()
    # print(g,d)
    optim_g = optim.Adam(g.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_d = optim.Adam(d.parameters(), lr=5e-4, betas=(0.5, 0.9))
    for epoch in range(100000):
        # train D net
        for _ in range(5):
            # train on real data
            x = next(data_iter)
            x = torch.from_numpy(x)
            predr = d(x)
            # max predr = min lossr
            lossr = - predr.mean()
            # train on fake data
            z = torch.randn(batch_size, 2)
            fake_data = g(z).detach()
            lossf = d(fake_data).mean()
            # gradient penalty
            gp = gradient_penalty(d, x, fake_data.detach())

            loss_d = lossr + lossf + gp * 0.2
            # optimze
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

        # train G net
        z = torch.randn(batch_size, 2)
        generater_data = g(z)
        g_loss = - d(generater_data).mean()
        optim_g.zero_grad()
        optim_d.zero_grad()
        g_loss.backward()
        optim_g.step()

        if epoch % 10 == 0:
            print(loss_d.item(), g_loss.item())
            vis.line([[loss_d.item(), g_loss.item()]], [epoch],
                     win="loss", update="append")

if __name__ == '__main__':
    main()
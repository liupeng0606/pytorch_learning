import torch

from torch import nn

a = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]], requires_grad=True)

b = torch.tensor([[7.0],[8.0],[9.0]], requires_grad=True)

c = a @ b

c = c.sum()

c.backward()

print(a.grad)
print(b.grad)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils as utils

epochs = 10000


transform=transforms.Compose([transforms.ToTensor()])


train_db = datasets.MNIST("./", train=True, download=True,
    transform = transform)

# transforms.Normalize((0.137,),(0.3081)
test_db = datasets.MNIST("./", train=False, download=True,
    transform=transform)

print(len(test_db))
print(len(train_db))


train_loader = utils.data.DataLoader(train_db, batch_size=128, shuffle = True)
test_loader = utils.data.DataLoader(test_db, shuffle = True)

print(train_loader)



class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.model(x)

net = MLP()

optimizer = optim.Adam(net.parameters(), lr=0.3)
criteon = nn.CrossEntropyLoss()





for epoch in range(1000):
    sum_loss = 0.0
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.view(-1,28*28)
        optimizer.zero_grad()
        logits = net(data)
        loss = criteon(logits,target)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if batch_index % 100 == 99:
            print('第%d个epoch的第%d个batch的loss: %.03f'
                  % (epoch, batch_index + 1, sum_loss / 100))
            sum_loss = 0.0
    print("**************************************************")
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.
        for data in test_loader:
            images, labels = data
            images = images.view(-1, 28 * 28)
            outputs = net(images)
            # 取得分最高的那个类
            test_loss += criteon(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum()
        print("\n")
        test_loss /= total
        print('第%d个epoch的识别准确率为：%.4f, test_loss is: %.4f' % (epoch, (100 * correct / total), test_loss))
        print("\n")



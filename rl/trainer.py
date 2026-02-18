import torch
import torch.optim as optim
from rl.agent import Net

def train_deep(steps=500):
    net = Net()
    opt = optim.Adam(net.parameters(), lr=0.01)

    total=0
    for i in range(steps):
        x = torch.randn(1,1)
        target = x*2
        pred = net(x)
        loss = ((pred-target)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    return round(total,4)

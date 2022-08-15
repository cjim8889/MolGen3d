import torch
from models.argmax.stochastic import NodeWiseStochasticPermutation
import time

batch_size = 1
mask = torch.ones(batch_size, 29, 1).to(torch.bool)
mask_size = torch.randint(3, 29, (batch_size, ))

for idx in range(batch_size):
    mask[idx, mask_size[idx]:] = False

x = torch.randn(batch_size, 29)
pos = torch.randn(batch_size, 29, 3)

pos.masked_fill_(~mask, 0)
x.masked_fill_(~mask.squeeze(2), 0.)

net = NodeWiseStochasticPermutation()


print(x, pos)
z, _ = net(x, pos, mask.squeeze(2))
print(x, pos)

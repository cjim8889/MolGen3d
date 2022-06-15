from models import CoorFlow
from models.argmax import AtomFlow

from pprint import pprint
import torch
from torch import nn





net = AtomFlow(hidden_dim=16, block_size=1, num_classes=6)


initial_mask = torch.ones(16, 29, dtype=torch.bool)
initial_mask[:, -1] = False

x = torch.randint(0, 6, (16, 29))
coord = torch.randn(16, 29, 3)

print(x[0])
z, _ = net(x, context=coord, mask=initial_mask)

x_, _ = net.inverse(z, context=coord, mask=initial_mask)

print(x_[0])
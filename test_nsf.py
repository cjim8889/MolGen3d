import torch
import numpy as np
from models.nsf import SplineCoorFlow

net = SplineCoorFlow(
    hidden_dim=64,
    block_size=12,
    max_nodes=18,
)


x = torch.randn(128, 18, 3)


z, _ = net(x)
print(z.shape)
print(sum([p.numel() for p in net.parameters()]))
from models import CoorFlow

from pprint import pprint
import torch
from torch import nn




initial_mask = torch.ones(16, 29, dtype=torch.bool)
initial_mask[:, -1] = False

net = CoorFlow(hidden_dim=32, gnn_size=1, block_size=1, max_nodes=29)

x = torch.randn(16, 29, 3)

z, _ = net(x, mask=initial_mask)

pprint(z.shape)
pprint(x[0])
pprint(z[0])
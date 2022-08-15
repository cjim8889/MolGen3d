from models.coordinates import CoorFlow
import torch

net = CoorFlow(
    hidden_dim=32,
    gnn_size=2,
    block_size=4,
    last_dimension=3
)

mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False


x = torch.randn(1, 29, 3)

z, _ = net(x, mask=mask)

print(z)

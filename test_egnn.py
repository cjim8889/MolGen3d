import torch
from models.egnn import ModifiedPosEGNN
from models import CoorFlow


# net = ModifiedPosEGNN(
#     in_dim = 3,
#     out_dim = 6,
#     m_dim=64,
#     fourier_features = 2,
#     num_nearest_neighbors = 6,
#     # dropout = 0.1,
#     norm_coors=True,
#     soft_edges=True
# )

net = CoorFlow(
    hidden_dim=32,
    gnn_size=2,
    block_size=2,
)

coors = torch.randn(1, 29, 3)
mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

print(coors)
out, _ = net(coors, mask=mask)
print(out)
print(out.shape)
from models import create_mask_equivariant, AdjacencyBlockFlow
from egnn_pytorch import EGNN
import torch
from torch import nn

from models.coordinates import CoorFlow


# mask = create_mask_equivariant(1, 3)

# print(mask)


if __name__ == "__main__":

    # net = EGNN(dim=6)

    net = CoorFlow(hidden_dim=32, gnn_size=1, block_size=4)

    # feats = torch.zeros(1, 9, 6)
    coors = torch.randn(1, 9, 3)

    coors = net(coors)
    print(coors)
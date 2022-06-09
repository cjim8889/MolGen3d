from models import create_mask_equivariant, create_mask_ar
from egnn_pytorch import EGNN
import torch
from torch import nn
from survae.utils import sum_except_batch
import numpy as np
from models.argmax.argmax import ContextNet, AtomFlow

from models.coordinates import CoorFlow

def assert_mean_zero_with_mask(x, node_mask):
    assert_correctly_masked(x, node_mask)
    assert torch.sum(x, dim=1, keepdim=True).abs().max().item() < 1e-4, \
        'Mean is not zero'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'
# mask = create_mask_equivariant(1, 3)
def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x
# print(mask)
def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    # assert len(x.size()) == 3
    node_mask = node_mask.unsqueeze(2)
    B, N_embedded, D = x.size()

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


if __name__ == "__main__":

    # net = EGNN(dim=6)

    # net = CoorFlow(hidden_dim=32, gnn_size=1, block_size=4)
    # net = ContextNet(hidden_dim=32, num_classes=5)
    net = AtomFlow(hidden_dim=32)

    feats = torch.randint(0, 5, size=(1, 9))
    coors = torch.randn(1, 9, 3)
    # coors = torch.randint(0, 5, size=(1, 9, 5))
    mask = create_mask_ar(26, (9, 3))
    mask = torch.ones(1, 9)
    mask[:, -1] = 0.
    mask = mask.to(torch.bool)

    z, _ = net(feats, coors, mask=mask)
    x, _ = net.inverse(z, coors, mask=mask)
    print(z, x, feats)
    # feats = torch.randn(1, 9, 6)
    # coors = torch.randn(1, 9, 3)
    # mask = create_mask_ar(26, (9, 3))
    # mask = torch.ones(1, 9)
    # mask[:, -1] = 0.

    # mask = mask.to(torch.bool)


    # out = net(feats, coors, mask=mask)
    # x = remove_mean_with_mask(coors, mask)
    # print(feats)
    # feats = net(feats)
    # print(feats)


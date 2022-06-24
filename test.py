import torch
from survae.utils import sum_except_batch
import numpy as np

from models.coordinates import CoorFlow

def assert_mean_zero_with_mask(x, node_mask):
    assert_correctly_masked(x, node_mask)
    assert torch.sum(x, dim=1, keepdim=True).abs().max().item() < 1e-4, \
        'Mean is not zero'


def assert_correctly_masked(variable, node_mask):
    assert (variable * ~ node_mask).abs().max().item() < 1e-4, \
        'Variables not masked properly.'
# mask = create_mask_equivariant(1, 3)
def remove_mean_with_mask(x, node_mask):
    assert (x * ~ node_mask.unsqueeze(2)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x
# print(mask)
def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
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

@torch.jit.script
def argmax_criterion(log_prob, log_det):
    return - torch.mean(log_prob + log_det)

if __name__ == "__main__":

    # net = EGNN(dim=6)

    net = CoorFlow(hidden_dim=128, gnn_size=2, block_size=8)

    pt = torch.load("model_irregularity_1426rlyj_400_81209.pt", map_location="cpu")

    net.load_state_dict(pt["model_state_dict"])

    input = pt['input']
    mask = pt['mask']

    z, log_det = net(input, mask=mask)

    z = z * mask.unsqueeze(2)
    zero_mean_z = remove_mean_with_mask(z, node_mask=mask)
    log_prob = sum_except_batch(center_gravity_zero_gaussian_log_likelihood_with_mask(zero_mean_z, node_mask=mask))

    loss = argmax_criterion(log_prob, log_det)    
    print(loss)
    # # net = ContextNet(hidden_dim=32, num_classes=5)
    # # net = AtomFlow(hidden_dim=32)

    # # feats = torch.randint(0, 5, size=(1, 9))
    # coors = torch.randn(1, 29, 3)
    # mask = torch.ones(1, 29)
    # mask[:, -1] = 0.
    # mask = mask.to(torch.bool)

    # z, _ = net(coors, mask=mask)

    

    # z, _ = net(feats, coors, mask=mask)
    # x, _ = net.inverse(z, coors, mask=mask)
    # print(z, x, feats)

    # mask = create_mask_equivariant(
    #     idx=[0, 1, 2], max_size=29
    # )


    # print(mask)
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


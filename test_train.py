import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from survae.utils import sum_except_batch

def remove_mean_with_constraint(x, size_constraint):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    # node_mask = node_mask.unsqueeze(2)s

    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x


def center_gravity_zero_gaussian_log_likelihood_with_constraint(x, size_constraint):
    # assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    # assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    # N = torch.full((x.shape[0]), size_constraint, device=device)
    degrees_of_freedom = (size_constraint - 1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px

x = torch.randn(16, 18, 3)

remove_mean_with_constraint(x, 18)

center_gravity_zero_gaussian_log_likelihood_with_constraint(x, 18)
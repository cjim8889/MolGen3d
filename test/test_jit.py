import torch
import time
def remove_mean_with_constraint(x, size_constraint):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    # node_mask = node_mask.unsqueeze(2)s
    # N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x

@torch.jit.script
def remove_mean_with_constraint_jit(x, size_constraint):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x



x = torch.randn(128, 18, 3)


remove_mean_with_constraint_jit(x, 18)
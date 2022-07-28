from models.egnn.resflow import Residual, ResCoorFlow
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from einops.layers.torch import Rearrange


residual = ResCoorFlow(hidden_dim=64, block_size=6)
x = torch.randn(1, 29, 3)
mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

x = x * mask.unsqueeze(2)
# print(x)
z, log_p = residual(x, mask=mask)
print(z)

x_r, _ = residual.inverse(z, mask=mask)

print(x, x_r)
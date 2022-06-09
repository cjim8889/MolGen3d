from survae.transforms.bijections import Bijection
from survae.utils import sum_except_batch

import torch
from torch import nn

class MaskedCouplingFlow(Bijection):
    def __init__(self, ar_net, mask, last_dimension=3, split_dim=1):
        super(MaskedCouplingFlow, self).__init__()
        
        self.ar_net = ar_net
        self.split_dim = split_dim
        
        self.register_buffer("mask", mask)
        self.scaling_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        return self._transform(x, mask=mask, forward=True)

    def inverse(self, z, mask=None):
        return self._transform(z, mask=mask, forward=False)

    def _transform(self, z, mask=None, forward=True):

        z_masked = z * self.mask

        alpha, beta = self.ar_net(z_masked, mask=mask).chunk(2, dim=self.split_dim)

        # scaling factor idea inspired by UvA github to stabilise training 
        scaling_factor = self.scaling_factor.exp().view(1, 1, 1)
        alpha = torch.tanh(alpha / scaling_factor) * scaling_factor

        alpha = alpha * (1 - self.mask)
        beta = beta * (1 - self.mask)
        
        if mask is not None:
            mask = mask.to(torch.float).unsqueeze(2)
            alpha = alpha * mask
            beta = beta * mask

        if forward:
            z = (z + beta) * torch.exp(alpha) # Exp to ensure invertibility
            log_det = sum_except_batch(alpha)
        else:
            z = (z * torch.exp(-alpha)) - beta
            log_det = -sum_except_batch(alpha)
        
        return z, log_det

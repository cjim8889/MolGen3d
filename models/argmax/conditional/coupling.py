from survae.transforms.bijections import ConditionalBijection
from survae.utils import sum_except_batch

from torch import nn
import torch

class MaskedConditionalCouplingFlow(ConditionalBijection):
    def __init__(self, ar_net, mask, split_dim=1, last_dimension=6):
        super(MaskedConditionalCouplingFlow, self).__init__()
        
        self.ar_net = ar_net
        self.register_buffer("mask", mask.unsqueeze(2))
        self.scaling_factor = nn.Parameter(torch.zeros(last_dimension))
        self.split_dim = split_dim

    def forward(self, x, context):
        return self._transform(x, context, forward=True)

    def inverse(self, z, context):
        return self._transform(z, context, forward=False)

    def _transform(self, z, context, forward=True):
        z_masked = z * self.mask
        alpha, beta = self.ar_net(z_masked, context).chunk(2, dim=self.split_dim)

        alpha = alpha.squeeze(1)
        beta = beta.squeeze(1)

        # scaling factor idea inspired by UvA github to stabilise training 
        scaling_factor = self.scaling_factor.exp().view(1, 1, -1)
        alpha = torch.tanh(alpha / scaling_factor) * scaling_factor

        alpha = alpha * ~self.mask
        beta = beta * ~self.mask
        
        if forward:
            z = (z + beta) * torch.exp(alpha) # Exp to ensure invertibility
            log_det = sum_except_batch(alpha)
        else:
            z = (z * torch.exp(-alpha)) - beta
            log_det = -sum_except_batch(alpha)
        
        return z, log_det
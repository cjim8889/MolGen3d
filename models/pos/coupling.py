from audioop import reverse
from survae.transforms.bijections import Bijection
from survae.utils import sum_except_batch
import torch.nn as nn
import torch

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

class AffineCouplingFlow(Bijection):
    def __init__(self, 
        ar_net, 
        scaling_func=lambda x: torch.exp(x), 
        split_dim=1,
        chunk_dim=2,
        n_dim=3,
        reverse=False,
    ):
        super(AffineCouplingFlow, self).__init__()
        
        self.ar_net = ar_net
        self.split_dim = split_dim
        self.chunk_dim = chunk_dim
        self.scaling_func = scaling_func
        self.reverse = reverse
    
    def _split(self, z):
        if self.reverse:
            return torch.chunk(
                z,
                2,
                dim=self.chunk_dim,
            )[::-1]
        else:
            return torch.chunk(
                z,
                2,
                dim=self.chunk_dim,
            )
    def _cat(self, z1, z2):
        if self.reverse:
            return torch.cat([z2, z1], dim=self.chunk_dim)
        else:
            return torch.cat([z1, z2], dim=self.chunk_dim)

    def _forward(self, x, forward=True):
        z1, z2 = self._split(x)
    
        params = self.ar_net(z1)
        z2, log_det = self._transform(z2, params, forward=forward)

        z = self._cat(z1, z2)

        return z, log_det

    def forward(self, x, mask=None, logs=None):
        return self._forward(x, forward=True)

    def inverse(self, x, mask=None):
        return self._forward(x, forward=False)

    def _transform(self, z, params, forward=True):
        alpha, shift = params.chunk(2, dim=self.split_dim)

        constrained_scale = self.scaling_func(alpha)

        if forward:
            z = z * constrained_scale + shift
            log_det = sum_except_batch(torch.log(constrained_scale))
        else:
            z = (z - shift) / constrained_scale
            log_det = -sum_except_batch(torch.log(constrained_scale))
        
        return z, log_det


class MaskedAffineCouplingFlow(Bijection):
    def __init__(self, 
        ar_net, 
        mask, 
        scaling_func=lambda x: torch.exp(x), 
        split_dim=1,
        n_dim=3
    ):
        super(MaskedAffineCouplingFlow, self).__init__()
        
        self.ar_net = ar_net
        self.split_dim = split_dim
        
        self.register_buffer("coupling_mask", mask)
        self.scaling_func = scaling_func

        self.scaling_factor = nn.Parameter(torch.zeros(n_dim))

    def forward(self, x, mask=None, logs=None):
        return self._transform(x, mask=mask, forward=True, logs=logs)

    def inverse(self, z, mask=None, logs=None):
        return self._transform(z, mask=mask, forward=False)

    
    def _transform(self, z, mask=None, forward=True, logs=None):
        z_masked = z * self.coupling_mask
        alpha, shift = self.ar_net(z_masked).chunk(2, dim=self.split_dim)

        scaling_factor = self.scaling_factor.exp().view(1, -1, 1)

        # constrained_scale = self.scaling_func(alpha)
        constrained_scale = torch.tanh(alpha / scaling_factor) * scaling_factor

        shift = shift * ~self.coupling_mask
        
        constrained_scale = constrained_scale * ~self.coupling_mask
        # constrained_scale.masked_fill_(self.coupling_mask, 1.)

        if mask is not None:
            shift = shift * mask
            constrained_scale = constrained_scale * mask

        if forward:
            z = (z + shift) * constrained_scale.exp()
            log_det = sum_except_batch(constrained_scale)
        else:
            z = (z * torch.exp(-constrained_scale)) - shift
            log_det = -sum_except_batch(constrained_scale)
        
        if logs is not None:
            logs.append(z.detach())

        return z, log_det

from .residual_flows.layers import iResBlock
import torch
from torch import nn
from survae.transforms.bijections import ConditionalBijection, Bijection
from torch.nn.utils.parametrizations import spectral_norm
from einops.layers.torch import Rearrange

class LipSwish_(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.swish = nn.SiLU(True)

    def forward(self, x):
        return self.swish(x).div_(1.1)

class Dynamics(nn.Module):
    def __init__(self, hidden_dim=64) -> None:
        super().__init__()

        # B X 29 X dim

        self.net = nn.Sequential(
            Rearrange(" b c d -> b () c d"),
            spectral_norm(nn.Conv2d(1, hidden_dim, kernel_size=3, stride=1, padding=1)),
            LipSwish_(),
            spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)),
            LipSwish_(),
            spectral_norm(nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)),
            Rearrange(" b () c d -> b c d"),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)

        return z

class Residual(Bijection):
    """
    Invertible residual net block, wrapper to the implementation of Chen et al.,
    see https://github.com/rtqichen/residual-flows
    """
    def __init__(self, net, n_exact_terms=2, n_samples=1, reduce_memory=True,
                 reverse=True):
        """
        Constructor
        :param net: Neural network, must be Lipschitz continuous with L < 1
        :param n_exact_terms: Number of terms always included in the power series
        :param n_samples: Number of samples used to estimate power series
        :param reduce_memory: Flag, if true Neumann series and precomputations
        for backward pass in forward pass are done
        :param reverse: Flag, if true the map f(x) = x + net(x) is applied in
        the inverse pass, otherwise it is done in forward
        """
        super().__init__()
        self.reverse = reverse
        self.iresblock = iResBlock(net, n_samples=n_samples,
                                   n_exact_terms=n_exact_terms,
                                   neumann_grad=reduce_memory,
                                   grad_in_forward=reduce_memory)

    def forward(self, z, mask=None, logs=None):
        if self.reverse:
            z, log_det = self.iresblock.inverse(z, 0, mask)
        else:
            z, log_det = self.iresblock.forward(z, 0, mask)
        
        if mask is not None:
            z = z * mask.unsqueeze(2)

        return z, -log_det.view(-1)

    def inverse(self, z, mask=None, logs=None):
        if self.reverse:
            z, log_det = self.iresblock.forward(z, 0, mask)
        else:
            z, log_det = self.iresblock.inverse(z, 0, mask)
        
        if mask is not None:
            z = z * mask.unsqueeze(2)

        return z, -log_det.view(-1)

class ResCoorFlow(nn.Module):
    def __init__(self, 
        hidden_dim=64, 
        block_size=6,
        max_nodes=29) -> None:

        super().__init__()

        self.transforms = nn.ModuleList([])

        for _ in range(block_size):
            net = Dynamics(hidden_dim)

            block = Residual(
                net,
                reduce_memory=True,
            )

            self.transforms.append(block)


    def forward(self, x, context=None, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)

            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context=None, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):

            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context, mask=mask)
            elif isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob
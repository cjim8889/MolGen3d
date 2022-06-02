from torch import nn
import torch
from survae.transforms.bijections import ConditionalBijection, Bijection

from .block import ar_net_init, AdjacencyBlockFlow

class CoorFlow(nn.Module):
    def __init__(self, 
        hidden_dim=64, 
        gnn_size=2,
        block_size=6) -> None:

        super().__init__()

        self.transforms = nn.ModuleList([AdjacencyBlockFlow(last_dimension=3, ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=gnn_size)) for _ in range(block_size)])

    def forward(self, x, context=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x)

            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):

            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context)
            elif isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z)

            log_prob += ldj
        
        return z, log_prob
from torch import nn
import torch
from survae.transforms.bijections import ConditionalBijection, Bijection

from .block import ar_net_init, CouplingBlockFlow

class CoorFlow(nn.Module):
    def __init__(self, 
        hidden_dim=64, 
        gnn_size=1,
        block_size=6,
        max_nodes=29) -> None:

        super().__init__()

        self.transforms = nn.ModuleList([CouplingBlockFlow(last_dimension=3, max_nodes=max_nodes, ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=gnn_size)) for _ in range(block_size)])

        for idx in range(1):
            tr = CouplingBlockFlow(
                last_dimension=3,
                ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=gnn_size),
                max_nodes=max_nodes,
                partition_size=1
            )

            self.transforms.append(tr)

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
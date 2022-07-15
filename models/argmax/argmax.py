from numpy import block
from torch import nn
import torch
from egnn_pytorch import EGNN
from .conditional import ConditionalAdjacencyBlockFlow
from .conditional.block import ar_net_init as ar_net_init_conditional

from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
from .surjectives import ArgmaxSurjection
from .block import ConditionalCouplingBlockFlow
from .block import ar_net_init as ar_net_init_block


class ContextNet(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=6) -> None:
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.embedding(x)
        return z

class AtomFlow(nn.Module):
    def __init__(self, 
        hidden_dim=32,
        block_size=2,
        num_classes=6
        ):

        super().__init__()

        self.transforms = nn.ModuleList()

        context_net = ContextNet(hidden_dim=hidden_dim, num_classes=num_classes)

        encoder_base = ConditionalNormal(
            nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes * 2),
            ),
            split_dim=-1
        )

        transforms = [
            ConditionalAdjacencyBlockFlow(
                max_nodes=29,
                num_classes=num_classes,
                ar_net_init=ar_net_init_conditional(hidden_dim=hidden_dim),
            ) for _ in range(2)
        ]

        conditional_flow = ConditionalInverseFlow(
            base_dist=encoder_base,
            transforms=transforms,
            context_init=context_net
        )

        surjection = ArgmaxSurjection(
            encoder=conditional_flow,
            num_classes=num_classes
        )

        self.transforms.append(surjection)

        self.transforms += [ConditionalCouplingBlockFlow(
                max_nodes=29,
                num_classes=num_classes,
                ar_net_init=ar_net_init_block(hidden_dim=hidden_dim, gnn_size=1),
                partition_size=1
            ) for _ in range(1)]

        self.transforms += [ConditionalCouplingBlockFlow(
            max_nodes=29,
            num_classes=6,
            ar_net_init=ar_net_init_block(hidden_dim=hidden_dim, gnn_size=1),
            partition_size=2
        ) for _ in range(block_size)]
    
    def forward(self, x, context, mask=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalCouplingBlockFlow):
                x, ldj = transform(x, context=context, mask=mask)
            else:
                x, ldj = transform(x, mask=mask)

            log_prob += ldj

        return x, log_prob

    def inverse(self, z, context, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], ConditionalCouplingBlockFlow):
                z, ldj = self.transforms[idx].inverse(z, context=context, mask=mask)
            else:
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob
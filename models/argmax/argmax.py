from torch import nn
import torch
from egnn_pytorch import EGNN
from .conditional import ConditionalAdjacencyBlockFlow

from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
from .surjectives import ArgmaxSurjection
from .block import ConditionalCouplingBlockFlow


class ContextNet(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=5) -> None:
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
                max_nodes=9,
                num_classes=num_classes
            )
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

        for i in range(block_size):
            flow = ConditionalCouplingBlockFlow(
                max_nodes=9,
                num_classes=num_classes
            )

            self.transforms.append(flow)
    
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
from .surjectives import ArgmaxSurjection
from survae.transforms.bijections import Bijection
from .conditionals import ConditionalBlockFlow, ar_net_init, MaskedConditionalNormal, MaskedConditionalInverseFlow
from .block import CouplingBlockFlow
from .block import ar_net_init as ar_net_init_block
from .stochastic import NodeWiseStochasticPermutation
import torch
from torch import nn

class ContextNet(nn.Module):
    def __init__(self, hidden_dim=64, context_dim=16, num_classes=6) -> None:
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, x, mask=None):
        z = self.embedding(x)

        if mask is not None:
            z *= mask.unsqueeze(2)
        
        return z

class AtomFlow(Bijection):
    def __init__(self, 
        num_classes=6, 
        hidden_dim=64, 
        gnn_size=2, 
        encoder_size=2,
        block_size=4,
        context_dim=16,
        no_constraint=False,
        euclidean_dim=3,
        max_nodes=29,
        stochastic_permute=True
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.euclidean_dim = euclidean_dim

        context_net = ContextNet(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_classes=num_classes
        )

        encoder_base = MaskedConditionalNormal(
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes * 2),
            ),
            split_dim=-1
        )

        transforms = [
            ConditionalBlockFlow(
                max_nodes=max_nodes,
                num_classes=num_classes,
                partition_size=2,
                no_constraint=no_constraint,
                ar_net_init=ar_net_init(hidden_dim=hidden_dim, num_classes=num_classes, gnn_size=gnn_size, context_dim=context_dim),
            ) for _ in range(encoder_size)
        ]



        inverse = MaskedConditionalInverseFlow(
            encoder_base, 
            transforms=transforms,
            context_init=context_net
        )

        surjection = ArgmaxSurjection(inverse, num_classes=num_classes)

        self.transforms = nn.ModuleList()

        if stochastic_permute:
            self.transforms.append(NodeWiseStochasticPermutation())

        self.transforms.append(surjection)


        self.transforms += [
            CouplingBlockFlow(
                num_classes=num_classes,
                euclidean_dim=3,
                ar_net_init=ar_net_init_block(hidden_dim=hidden_dim, num_classes=num_classes, gnn_size=gnn_size),
                no_constraint=no_constraint,
                partition_size=2,
            ) for _ in range(block_size)
        ]


    def forward(self, x, pos, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ArgmaxSurjection):
                x, ldj = transform.forward(x, mask=mask)
            elif isinstance(transform, Bijection) or isinstance(transform, NodeWiseStochasticPermutation):
                x, ldj = transform(x, pos, mask=mask, logs=logs)

            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, pos, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], ArgmaxSurjection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)
            elif isinstance(self.transforms[idx], Bijection) or isinstance(self.transforms[idx], NodeWiseStochasticPermutation):
                z, ldj = self.transforms[idx].inverse(z, pos, mask=mask)

            log_prob += ldj
        
        return z, log_prob
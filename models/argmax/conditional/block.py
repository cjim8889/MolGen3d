from survae.transforms.bijections import ConditionalBijection
import torch
from torch import nn

from .coupling import MaskedConditionalCouplingFlow

def create_mask_equivariant(idx, max_size, invert=False):

    mask = torch.zeros(max_size, dtype=torch.float32)

    mask[idx] = 1
    mask = mask.view(1, max_size)

    if not invert:
        mask = 1 - mask

    return mask.to(torch.bool)

class ConditionalARNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, 3, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(2, 1, 1, 0),
            nn.ReLU(),
        )

    # x: B x 9 x 6
    # context: B x 9 x 6
    def forward(self, x, context):
        z = torch.stack((x, context), dim=1)
        z = self.net(z)

        return z

def ar_net_init(**kwargs):
    def create():
        return ConditionalARNet(**kwargs)

    return create

class ConditionalAdjacencyBlockFlow(ConditionalBijection):
    def __init__(self, ar_net_init=ar_net_init(hidden_dim=64),
            max_nodes=9,
            num_classes=6,
            mask_init=create_mask_equivariant,
            split_dim=1):

        super(ConditionalAdjacencyBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(max_nodes):
            ar_net = ar_net_init()
            mask = mask_init(idx, max_nodes)

            tr = MaskedConditionalCouplingFlow(ar_net, mask=mask, split_dim=split_dim)
            self.transforms.append(tr)
        
       
    def forward(self, x, context):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            x, ldj = transform(x, context)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context):
        log_prob = torch.zeros(z.shape[0], device=z.device)
        for idx in range(len(self.transforms) - 1, -1, -1):
            z, ldj = self.transforms[idx].inverse(z, context)
            log_prob += ldj
        
        return z, log_prob


from survae.transforms.bijections import Bijection
from .utils import create_mask_equivariant
from .coupling import MaskedCouplingFlow
from survae.transforms.bijections import ConditionalBijection, Bijection

import torch
from torch import nn
from egnn_pytorch import EGNN

class ARNet(nn.Module):
    def __init__(self, hidden_dim=128, gnn_size=2):
        super().__init__()

        self.net = nn.ModuleList([EGNN(dim=6, m_dim=hidden_dim, norm_coors=True, soft_edges=True) for _ in range(gnn_size)])

    def forward(self, x):

        feats = torch.zeros(x.shape[0], x.shape[1], 6, device=x.device)
        coors = x

        for net in self.net:
            feats, coors = net(feats, coors)
        
        return feats

def ar_net_init(hidden_dim=128, gnn_size=2):
    def _init():
        return ARNet(hidden_dim=hidden_dim, gnn_size=gnn_size)

    return _init

class AdjacencyBlockFlow(Bijection):
    def __init__(self,
    last_dimension=3,
    ar_net_init=ar_net_init(hidden_dim=32, gnn_size=2),
    mask_init=create_mask_equivariant):
        
        super(AdjacencyBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(3):
            ar_net = ar_net_init()
            mask = mask_init(idx, 3)

            tr = MaskedCouplingFlow(ar_net, mask=mask, last_dimension=last_dimension, split_dim=-1)
            self.transforms.append(tr)
        
    
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
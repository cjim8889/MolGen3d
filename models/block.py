from survae.transforms.bijections import Bijection
from .utils import create_mask_ar
from .coupling import MaskedCouplingFlow
from survae.transforms.bijections import ConditionalBijection, Bijection, ActNormBijection1d

import torch
from torch import nn
from egnn_pytorch import EGNN

class ARNet(nn.Module):
    def __init__(self, hidden_dim=128, gnn_size=2):
        super().__init__()

        self.net = nn.ModuleList([EGNN(dim=6, m_dim=hidden_dim, norm_coors=True, soft_edges=True, coor_weights_clamp_value=1., update_coors=False, num_nearest_neighbors=6) for _ in range(gnn_size)])

    def forward(self, x, mask=None):
        feats = x.repeat(1, 1, 2)
        coors = x

        for net in self.net:
            feats, coors = net(feats, coors, mask=mask)
        
        return feats

def ar_net_init(hidden_dim=128, gnn_size=2):
    def _init():
        return ARNet(hidden_dim=hidden_dim, gnn_size=gnn_size)

    return _init

class CouplingBlockFlow(Bijection):
    def __init__(self,
    last_dimension=3,
    ar_net_init=ar_net_init(hidden_dim=32, gnn_size=2),
    mask_init=create_mask_ar,
    max_nodes=29):
        
        super(CouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(max_nodes * last_dimension):
            ar_net = ar_net_init()
            mask = mask_init(idx, (max_nodes, last_dimension))

            tr = MaskedCouplingFlow(ar_net, mask=mask, last_dimension=last_dimension, split_dim=-1)
            self.transforms.append(tr)
        
    
    def forward(self, x, context=None, mask=None,):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask)
            elif isinstance(transform, ActNormBijection1d):
                x, ldj = transform(x)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask)

            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context=None, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):

            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context, mask=mask)
            elif isinstance(self.transforms[idx], ActNormBijection1d):
                x, ldj = self.transform[idx].inverse(z)
            elif isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob
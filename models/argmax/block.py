from survae.transforms.bijections import Bijection
from ..utils import create_mask_equivariant
from .coupling import MaskedConditionalCouplingFlow
from survae.transforms.bijections import ConditionalBijection, Bijection

import torch
from torch import nn
from egnn_pytorch import EGNN

class ARNet(nn.Module):
    def __init__(self, hidden_dim=32, gnn_size=1, num_classes=6, idx=(0, 2)):
        super().__init__()

        self.idx = idx
        self.net = nn.ModuleList([EGNN(dim=num_classes*2, m_dim=hidden_dim, soft_edges=True, update_coors=False, num_nearest_neighbors=6) for _ in range(gnn_size)])

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear((self.idx[1] - self.idx[0]) * num_classes * 2)
        )
        
        self.num_classes = num_classes

    def forward(self, x, context=None, mask=None):
        feats = x.repeat(1, 1, 2)
        mask = mask.expand(x.shape[0], mask.shape[1])
        coors = context

        for net in self.net:
            feats, coors = net(feats, coors, mask=mask)
        
        feats = torch.sum(feats * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)
        feats = self.mlp(feats).view(x.shape[0], self.idx[1] - self.idx[0], self.num_classes * 2)
        feats = nn.functional.pad(feats, (0, 0, self.idx[0], 29 - self.idx[1], 0, 0), 'constant', 0)
        

        return feats

def ar_net_init(hidden_dim=128, gnn_size=1):
    def _init(idx):
        return ARNet(hidden_dim=hidden_dim, gnn_size=gnn_size, idx=idx)

    return _init

class ConditionalCouplingBlockFlow(ConditionalBijection):
    def __init__(self,
    max_nodes=29,
    num_classes=6,
    ar_net_init=ar_net_init(hidden_dim=64, gnn_size=1),
    mask_init=create_mask_equivariant,
    partition_size=2):
        
        super(ConditionalCouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, partition_size):
            ar_net = ar_net_init((idx, min(idx + partition_size, max_nodes)))
            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)

            tr = MaskedConditionalCouplingFlow(ar_net, mask=mask, split_dim=-1)
            self.transforms.append(tr)
        
    
    def forward(self, x, context=None, mask=None,):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask)

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
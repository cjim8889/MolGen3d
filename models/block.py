from survae.transforms.bijections import Bijection
from .utils import create_mask_ar, create_mask_equivariant
from .coupling import MaskedCouplingFlow
from survae.transforms.bijections import ConditionalBijection, Bijection, ActNormBijection1d

import torch
from torch import nn
from .egnn.egnn import ModifiedPosEGNN
from .actnorm import ActNorm

class ARNet(nn.Module):
    def __init__(self, hidden_dim=32, gnn_size=1, idx=(0, 2), activation='LipSwish'):
        super().__init__()

        self.idx = idx

        self.net = nn.ModuleList([ModifiedPosEGNN(out_dim=6, m_dim=hidden_dim, activation=activation, fourier_features=0, soft_edges=False, norm_coors=False) for _ in range(gnn_size)])

        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, (self.idx[1] - self.idx[0]) * 6),
            nn.Sigmoid()
        )

        self.eps = 1e-6
        
    def forward(self, x, mask=None):
        coors = x
        for net in self.net:
            coors = net(coors, mask=mask)
        
        coors = coors * mask.unsqueeze(2)
        # x = x * mask.unsqueeze(2)

        # x = torch.sum(x, dim=1) / (degree + self.eps)
        coors = torch.sum(coors, dim=1)
        
        # coors = self.mlp(torch.cat((x, coors), dim=-1)).view(x.shape[0], self.idx[1] - self.idx[0], 6)
        coors = self.mlp(coors).view(x.shape[0], self.idx[1] - self.idx[0], 6)
        coors = nn.functional.pad(coors, (0, 0, self.idx[0], 29 - self.idx[1], 0, 0), 'constant', 0)
        
        return coors

def ar_net_init(hidden_dim=128, gnn_size=2, activation='LipSwish'):
    def _init(idx):
        return ARNet(hidden_dim=hidden_dim, gnn_size=gnn_size, idx=idx, activation=activation)

    return _init

class CouplingBlockFlow(Bijection):
    def __init__(self,
    last_dimension=3,
    ar_net_init=ar_net_init(hidden_dim=64, gnn_size=1),
    mask_init=create_mask_equivariant,
    max_nodes=29,
    act_norm=True,
    partition_size=2):
        
        super(CouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, partition_size):
            if act_norm:
                norm = ActNorm(last_dimension)
                self.transforms.append(norm)
            
            ar_net = ar_net_init((idx, min(idx + partition_size, max_nodes)))
            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)
            tr = MaskedCouplingFlow(ar_net, mask=mask, last_dimension=last_dimension, split_dim=-1)
            self.transforms.append(tr)
    
    def forward(self, x, context=None, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask)
            elif isinstance(transform, ActNormBijection1d):
                x, ldj = transform(x)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)

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
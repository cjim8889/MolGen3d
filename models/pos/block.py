from survae.transforms.bijections import Bijection
from .utils import create_mask_dim_wise, create_mask_node_wise
from .coupling import MaskedAffineCouplingFlow
from survae.transforms.bijections import Bijection

from .transformer import DenseTransformer
import torch
from torch import nn

class CouplingBlockFlow(Bijection):
    def __init__(self) -> None:
        super(CouplingBlockFlow, self).__init__()

    def forward(self, x, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob

class DimWiseCouplingBlockFlow(CouplingBlockFlow):
    def __init__(self,
        n_dim=3,
        mask_init=create_mask_dim_wise,
        num_layers_transformer=6,
        hidden_dim=64,
        max_nodes=29,
        partition_size=1
    ):
        super(DimWiseCouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, n_dim, partition_size):
            ar_net = DenseTransformer(
                d_input=n_dim,
                d_output=n_dim * 2,
                d_model=hidden_dim,
                num_layers=num_layers_transformer,
                dim_feedforward=hidden_dim * 2,
                # dropout=0
            )

            mask = mask_init([i for i in range(idx, min(idx + partition_size, n_dim))], n_dim)
            
            tr = MaskedAffineCouplingFlow(
                ar_net, 
                mask=mask,
                scaling_func=torch.nn.Softplus(),
                split_dim=1
            )

            self.transforms.append(tr)


class NodeWiseCouplingBlockFlow(CouplingBlockFlow):
    def __init__(self,
        n_dim=3,
        mask_init=create_mask_node_wise,
        num_layers_transformer=6,
        hidden_dim=64,
        max_nodes=29,
        partition_size=2,
        start_idx=0
    ):
        
        super(NodeWiseCouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(start_idx, max_nodes, partition_size):
            ar_net = DenseTransformer(
                d_input=n_dim,
                d_output=n_dim * 2,
                d_model=hidden_dim,
                num_layers=num_layers_transformer,
                dim_feedforward=hidden_dim * 2,
                # dropout=0
            )

            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)
            
            tr = MaskedAffineCouplingFlow(
                ar_net, 
                mask=mask,
                scaling_func=torch.nn.Softplus(),
                split_dim=1
            )

            self.transforms.append(tr)
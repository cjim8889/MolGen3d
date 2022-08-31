from ast import arg
from turtle import forward
import torch
from torch import nn
from .transformer import DenseTransformer
from einops.layers.torch import Rearrange
from .utils import create_mask_dim_wise, create_mask_node_wise
from survae.transforms.bijections import ConditionalBijection, Bijection
from .coupling import ConditionalMaskedAffineCouplingFlow
from .conv import Conv1x1
from .norm import ActNormFlow

class AtomContextNet(nn.Module):
    def __init__(self,
        n_categories = 5,
        context_dim = 6,
        hidden_dim = 128,
        n_layers = 6,
        max_nodes = 29,
        dim_feedforward = 128,
        n_heads = 8,
    ):
        super(AtomContextNet, self).__init__()

        self.net = nn.Sequential(
            nn.Embedding(n_categories, context_dim),
            nn.Tanh(),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            Rearrange("b c k -> b k c"),
            DenseTransformer(d_input=context_dim, d_output=context_dim, d_model=hidden_dim, num_layers=n_layers, dim_feedforward=dim_feedforward, dropout=0, nhead=n_heads),
            # Rearrange("b k c -> b c k"),
            # DenseTransformer(d_input=max_nodes, d_output=max_nodes, d_model=hidden_dim, num_layers=n_layers, dim_feedforward=dim_feedforward, dropout=0, nhead=n_heads),
            # Rearrange("b c k -> b k c"),
        )
    
    '''
    Output Dim: B X Context_dim X Max_nodes
    '''
    def forward(self, x):
        return self.net(x)

class ConditionalCouplingBlockFlow(ConditionalBijection):
    def __init__(self) -> None:
        super(ConditionalCouplingBlockFlow, self).__init__()

    def forward(self, x, context, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask, logs=logs)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context, mask=mask)
            elif isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob

class ConditionalTransformer(DenseTransformer):
    def __init__(self, *args, **kwargs):
        super(ConditionalTransformer, self).__init__(*args, **kwargs)
    
    def forward(self, x, context):
        x = torch.cat([x, context], dim=1)

        return super().forward(x)


class ConditionalDimWiseCouplingBlockFlow(ConditionalCouplingBlockFlow):
    def __init__(self,
        n_dim=3,
        mask_init=create_mask_dim_wise,
        num_layers_transformer=6,
        hidden_dim=64,
        max_nodes=29,
        partition_size=1,
        act_norm=False,
        conv1x1=False,
        context_dim=6,
        dropout=0,
    ):
        super(ConditionalDimWiseCouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, n_dim, partition_size):
            ar_net = ConditionalTransformer(
                d_input=n_dim + context_dim,
                d_output=n_dim * 2,
                d_model=hidden_dim,
                num_layers=num_layers_transformer,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
            )

            mask = mask_init([i for i in range(idx, min(idx + partition_size, n_dim))], n_dim)
            
            tr = ConditionalMaskedAffineCouplingFlow(
                ar_net, 
                mask=mask,
                scaling_func=torch.nn.Softplus(),
                split_dim=1
            )

            if act_norm:
                self.transforms.append(ActNormFlow(
                    num_features=max_nodes,
                ))
            
            if conv1x1:
                self.transforms.append(
                    Conv1x1(
                        num_channels=max_nodes,
                        node_wise=True,
                    )
                )

            self.transforms.append(tr)


class ConditionalNodeWiseCouplingBlockFlow(ConditionalCouplingBlockFlow):
    def __init__(self,
        n_dim=3,
        mask_init=create_mask_node_wise,
        num_layers_transformer=6,
        hidden_dim=64,
        max_nodes=29,
        partition_size=2,
        start_idx=0,
        act_norm=False,
        conv1x1=False,
        dropout=0.1,
        context_dim=6,
    ):
        
        super(ConditionalNodeWiseCouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(start_idx, max_nodes, partition_size):
            ar_net = ConditionalTransformer(
                d_input=n_dim + context_dim,
                d_output=n_dim * 2,
                d_model=hidden_dim,
                num_layers=num_layers_transformer,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout
            )

            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)
            
            tr = ConditionalMaskedAffineCouplingFlow(
                ar_net, 
                mask=mask,
                scaling_func=torch.nn.Softplus(),
                split_dim=1
            )

            if act_norm:
                self.transforms.append(ActNormFlow(
                    num_features=max_nodes,
                ))
            
            if conv1x1:
                self.transforms.append(
                    Conv1x1(
                        num_channels=max_nodes,
                        node_wise=True,
                    )
                )

            self.transforms.append(tr)
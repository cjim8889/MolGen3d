from torch import nn
from .block import DimWiseCouplingBlockFlow, NodeWiseCouplingBlockFlow, CouplingBlockFlow
from .conditional import ConditionalDimWiseCouplingBlockFlow, ConditionalNodeWiseCouplingBlockFlow, AtomContextNet
from .conv import Conv1x1
from survae.transforms.bijections import ConditionalBijection, Bijection
from .norm import BatchNormFlow, ActNormFlow
import torch

class ConditionalTransformerCoorFlow(nn.Module):
    def __init__(self, 
            hidden_dim=64, 
            block_size=6,
            max_nodes=29,
            num_layers_transformer=6,
            partition_size=(1, 9),
            n_dim=3,
            n_categories=5,
            dim_wise=True,
            node_wise=True,
            conv1x1=True,
            conv1x1_node_wise=False,
            act_norm=True,
            squeeze=False,
            squeeze_step=3,
            context_dim=6
        ) -> None:

        super(ConditionalTransformerCoorFlow, self).__init__()

        self.transforms = nn.ModuleList()


        self.context_net = AtomContextNet(
            n_categories=n_categories,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers_transformer,
            max_nodes=max_nodes,
            dim_feedforward=hidden_dim * 2,
        )

        for idx in range(block_size):
            if dim_wise:
                self.transforms.append(
                    ConditionalDimWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[0],
                        act_norm=act_norm,
                        conv1x1=conv1x1,
                        dropout=0,
                    )
                )
            
            if node_wise:
                self.transforms.append(
                    ConditionalNodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[1],
                        act_norm=act_norm,
                        conv1x1=conv1x1,
                        dropout=0,
                    )
                )

        if squeeze:
            for start_idx in range(squeeze_step, max_nodes, squeeze_step):
                self.transforms.append(
                    ConditionalNodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[1],
                        start_idx=start_idx,
                        act_norm=act_norm,
                        conv1x1=conv1x1,
                        dropout=0,
                    )
                )
    
    def forward(self, x, context, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        if self.context_net is not None:
            context = self.context_net(context)

        for transform in self.transforms:
            if isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask, logs=logs)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        if self.context_net is not None:
            context = self.context_net(context)

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)
            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context, mask=mask)

            log_prob += ldj
        
        return z, log_prob

class TransformerCoorFlowV2(CouplingBlockFlow):
    def __init__(self, 
            hidden_dim=64, 
            block_size=6,
            max_nodes=29,
            num_layers_transformer=6,
            partition_size=(1, 9),
            n_dim=3,
            dim_wise=True,
            node_wise=True,
            conv1x1=True,
            conv1x1_node_wise=False,
            batch_norm=True,
            act_norm=True,
            squeeze=False,
            squeeze_step=3
        ) -> None:

        super(TransformerCoorFlowV2, self).__init__()

        self.transforms = nn.ModuleList()

        for idx in range(block_size):
            if dim_wise:
                self.transforms.append(
                    DimWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[0],
                        act_norm=act_norm,
                        conv1x1=conv1x1,
                        dropout=0,
                    )
                )
            
            if node_wise:
                self.transforms.append(
                    NodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[1],
                        act_norm=act_norm,
                        conv1x1=conv1x1,
                        dropout=0,
                    )
                )

        if squeeze:
            for start_idx in range(squeeze_step, max_nodes, squeeze_step):
                self.transforms.append(
                    NodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[1],
                        start_idx=start_idx,
                        act_norm=act_norm,
                        conv1x1=conv1x1,
                        dropout=0,
                    )
                )

class TransformerCoorFlow(CouplingBlockFlow):
    def __init__(self, 
            hidden_dim=64, 
            block_size=6,
            max_nodes=29,
            num_layers_transformer=6,
            partition_size=(1, 9),
            n_dim=3,
            dim_wise=True,
            node_wise=True,
            conv1x1=True,
            conv1x1_node_wise=False,
            batch_norm=True,
            act_norm=True,
            squeeze=False,
            squeeze_step=3
        ) -> None:

        super(TransformerCoorFlow, self).__init__()

        self.transforms = nn.ModuleList()

        for idx in range(block_size):
            if dim_wise:
                self.transforms.append(
                    DimWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[0],
                    )
                )
            
            if act_norm:
                self.transforms.append(ActNormFlow(
                    num_features=max_nodes,
                ))
            
            if conv1x1:
                self.transforms.append(
                    Conv1x1(
                        num_channels=max_nodes if conv1x1_node_wise else n_dim,
                        node_wise=conv1x1_node_wise,
                    )
                )

            if node_wise:
                self.transforms.append(
                    NodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[1]
                    )
                )
            
            if act_norm:
                self.transforms.append(ActNormFlow(
                    num_features=max_nodes,
                ))

            if batch_norm:
                self.transforms.append(
                    BatchNormFlow(
                        num_features=max_nodes,
                    )
                )

            if conv1x1:
                self.transforms.append(
                    Conv1x1(
                        num_channels=max_nodes if conv1x1_node_wise else n_dim,
                        node_wise=conv1x1_node_wise,
                    )
                )

        if squeeze:
            for start_idx in range(squeeze_step, max_nodes, squeeze_step):
                self.transforms.append(
                    NodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size[1],
                        start_idx=start_idx
                    )
                )

                if conv1x1:
                    self.transforms.append(
                        Conv1x1(
                            num_channels=max_nodes if conv1x1_node_wise else n_dim,
                            node_wise=conv1x1_node_wise,
                        )
                    )
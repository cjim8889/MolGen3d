from torch import nn
from .block import DimWiseCouplingBlockFlow, NodeWiseCouplingBlockFlow, CouplingBlockFlow
from .conv import Conv1x1
from .batchnorm import BatchNormFlow
class TransformerCoorFlow(CouplingBlockFlow):
    def __init__(self, 
            hidden_dim=64, 
            block_size=6,
            max_nodes=29,
            num_layers_transformer=6,
            partition_size=9,
            n_dim=3,
            dim_wise=True,
            node_wise=True,
            conv1x1=True,
            conv1x1_node_wise=False,
            batch_norm=True,
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
                        partition_size=1,
                    )
                )
            
            if node_wise:
                self.transforms.append(
                    NodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=partition_size
                    )
                )

            if batch_norm:
                self.transforms.append(
                    BatchNormFlow(
                        num_features=n_dim,
                    )
                )

            if conv1x1:
                self.transforms.append(
                    Conv1x1(
                        num_channels=max_nodes if conv1x1_node_wise else n_dim,
                        node_wise=conv1x1_node_wise,
                    )
                )

from torch import nn
from .block import DimWiseCouplingBlockFlow, NodeWiseCouplingBlockFlow, CouplingBlockFlow

class TransformerCoorFlow(CouplingBlockFlow):
    def __init__(self, 
            hidden_dim=64, 
            block_size=6,
            max_nodes=29,
            num_layers_transformer=6,
            n_dim=3,
            dim_wise=True,
            node_wise=True
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
                        partition_size=2
                    )
                )
            
            if node_wise:
                self.transforms.append(
                    NodeWiseCouplingBlockFlow(
                        n_dim=n_dim,
                        num_layers_transformer=num_layers_transformer,
                        hidden_dim=hidden_dim,
                        max_nodes=max_nodes,
                        partition_size=1
                    )
                )


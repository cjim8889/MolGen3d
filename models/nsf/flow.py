from torch import nn
from ..pos.conv import Conv1x1
from ..pos.block import CouplingBlockFlow
from .spline import SplineFlow

class SplineCoorFlow(CouplingBlockFlow):
    def __init__(self, 
            hidden_dim=64, 
            block_size=6,
            max_nodes=29,
            n_dim=3,
            conv1x1=False,
            conv1x1_node_wise=False,
        ) -> None:

        super(SplineCoorFlow, self).__init__()

        self.transforms = nn.ModuleList()

        for idx in range(block_size):
            self.transforms.append(
                SplineFlow(
                    n_dim=n_dim,
                    hidden_dim=hidden_dim,
                    max_nodes=max_nodes,
                    num_bins=12,
                    hidden_length=4
                )
            )

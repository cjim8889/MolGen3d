from torch import nn
import torch
from .transformer import DenseTransformer
from einops.layers.torch import Rearrange
class BaseNet(nn.Sequential):
    def __init__(self, hidden_dim=64, num_layers=4, max_nodes=18, n_dim=3) -> None:
        super(BaseNet, self).__init__()

        self.append(
            Rearrange("b (c d) -> b c d", c=n_dim, d=max_nodes),
        )

        self.append(
            DenseTransformer(
                d_input=n_dim,
                d_output=1,
                d_model=hidden_dim,
                dim_feedforward=hidden_dim * 2,
                num_layers=num_layers,
                dropout=0.1
            ),
        )

        self.append(
            nn.Sequential(
                Rearrange("b c d -> b (c d)"),
                nn.Linear(max_nodes * 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        )
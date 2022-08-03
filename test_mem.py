import torch.autograd.profiler as profiler
import torch
from models.pos.flow import TransformerCoorFlow
from utils import get_datasets



model = TransformerCoorFlow(
    hidden_dim=32,
    block_size=6,
    max_nodes=18,
    num_layers_transformer=4,
    partition_size=(1, 9),
    conv1x1_node_wise=True,
    batch_norm=True
)

input = torch.randn(128, 3, 18)

model(input)
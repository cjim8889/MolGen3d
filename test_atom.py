from models.pos.transformer import DenseTransformer
from models.pos.coupling import MaskedAffineCouplingFlow
from models.pos.block import NodeWiseCouplingBlockFlow, DimWiseCouplingBlockFlow
from models.pos.flow import TransformerCoorFlow
import torch




mask = torch.ones(1, 1, 29, dtype=torch.bool)
mask[:, :, -2:] = False

flow = TransformerCoorFlow(
    hidden_dim=32,
    block_size=2,
    max_nodes=29,
    num_layers_transformer=4,
    n_dim=3,
)

x = torch.randn(1, 3, 29)
x.masked_fill_(~mask, 0.)
print(x)
z, log_det = flow(x, mask=mask)

x_re, _ = flow.inverse(z)
print(x_re)

print(sum([p.numel() for p in flow.parameters()]))


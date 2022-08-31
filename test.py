from models.pos.conditional import AtomContextNet, ConditionalDimWiseCouplingBlockFlow
from models.pos.flow import ConditionalTransformerCoorFlow
import torch





net = ConditionalTransformerCoorFlow(
    hidden_dim=128,
    block_size=2,
    max_nodes=29,
    num_layers_transformer=6,
    partition_size=(1, 6),
    n_dim=3,
    n_categories=5,
    context_dim=6,
)

x = torch.randn(1, 3, 29)
context = torch.randint(0, 5, (1, 29))


out, _ = net(x, context)
print(out.shape)
from models.argmax.atom import AtomFlow
import torch



net = AtomFlow(
    num_classes=5,
    hidden_dim=16,
    gnn_size=2,
    encoder_size=2,
    block_size=2,
    context_dim=16,
)


mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

x = torch.randint(0, 5, (1, 29))
pos = torch.randn(1, 29, 3)
print(x)
z, _ = net.forward(x, pos, mask=mask)

x_re, _ = net.inverse(z, pos, mask=mask)
print(x_re)
print(z)
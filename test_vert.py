from models.argmax.atom import AtomFlow
import torch



net = AtomFlow(
    num_classes=5,
    hidden_dim=16,
    gnn_size=2,
    encoder_size=2,
    block_size=2,
    context_dim=16,
    stochastic_permute=True
)


mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

x = torch.randint(0, 5, (1, 29))
pos = torch.randn(1, 29, 3)

x.masked_fill_(~mask, 0)
pos.masked_fill_(~mask.unsqueeze(2), 0.)

print(pos)
z, _ = net.forward(x, pos, mask=mask)

x_re, _ = net.inverse(z, pos, mask=mask)
print(pos)
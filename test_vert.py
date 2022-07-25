from models.argmax.argmax import AtomFlow
import torch



net = AtomFlow(
    hidden_dim=16,
    block_size=2,
    num_classes=6
)


mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

out, _ = net.forward(torch.randint(0, 6, (1, 29)), torch.randn(1, 29, 3), mask=mask)

print(net)
print(out)
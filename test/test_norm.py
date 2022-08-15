from models.pos.norm import ActNormFlow
import torch




x = torch.randn(128, 3, 18)

net = ActNormFlow(
    num_features=18,
)



z, _ = net(x)

print(z.shape, z)
from models.pos.conv import Conv1x1
import torch


feats = torch.randn(1, 3, 29)

net = Conv1x1(
    num_channels=3,
)



z, _ = net(feats)
# print(z)

x, _ = net.inverse(z)
print(feats, x)
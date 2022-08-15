from models.pos.conv import Conv1x1
import torch
from models.pos.batchnorm import BatchNormFlow


feats = torch.randn(1, 3, 29)

net = BatchNormFlow(
    num_features=3
)



z, log_det = net(feats)
# print(z)

# x, _ = net.inverse(z)
print(feats, z)
print(feats.shape, z.shape, log_det.shape)
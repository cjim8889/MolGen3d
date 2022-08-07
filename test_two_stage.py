
import torch
from torch import nn

x = torch.rand(128, 29, 3)

output = torch.rand(128) < 0.5


print(x[output].shape, output)
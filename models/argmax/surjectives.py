from survae.transforms.surjections import Surjection
from einops.layers.torch import Rearrange
from survae.transforms import Softplus
from torch.nn import functional as F
from survae.utils import sum_except_batch
import torch
from torch import nn

class ContextNet(nn.Module):
    def __init__(self, hidden_dim=64, context_dim=16, num_classes=6) -> None:
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, x, mask=None):
        z = self.embedding(x)

        if mask is not None:
            z *= mask.unsqueeze(2)
        
        return z

class ArgmaxSurjection(Surjection):
    stochastic_forward = True

    def __init__(self, encoder, num_classes):
        super(ArgmaxSurjection, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.softplus = Softplus()
        self.rearrange = Rearrange("B W -> B W 1")

    
    def forward_old(self, x, mask=None):
        u, log_pu = self.encoder.sample_with_log_prob(context=x)
        
        index = self.rearrange(x)

        # Thresholding
        u_max = torch.take_along_dim(u, index, dim=-1)
        u_x = u_max - u

        u_tmp = F.softplus(u_x)
        ldj = F.logsigmoid(u_x)

        v = u_max - u_tmp

        ldj = ldj.scatter_(2, index, 0.)
        v = v.scatter_(2, index, u_max)

        log_pz = log_pu - torch.sum(ldj, dim=[1, 2])

        return v, -log_pz

    def forward(self, x, mask=None):
        u, log_pu = self.encoder.sample_with_log_prob(context=x, mask=mask)
        onehot = nn.functional.one_hot(x, num_classes=self.num_classes) * mask.unsqueeze(2)

        mask_ = mask.unsqueeze(2)

        T = (onehot * u).sum(-1, keepdim=True)
        z = onehot * u + mask_ * (1 - onehot) * (T - F.softplus(T - u))
        ldj = (1 - onehot) * F.logsigmoid(T - u) * mask_

        ldj = sum_except_batch(ldj)

        return z, -(log_pu - ldj)

    def inverse(self, z, mask=None):
        return z.argmax(dim=-1), 0.

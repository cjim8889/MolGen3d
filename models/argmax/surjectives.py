from survae.transforms.surjections import Surjection
from einops.layers.torch import Rearrange
from survae.transforms import Softplus
from torch.nn import functional as F
import torch

class ArgmaxSurjection(Surjection):
    stochastic_forward = True

    def __init__(self, encoder, num_classes):
        super(ArgmaxSurjection, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.softplus = Softplus()
        self.rearrange = Rearrange("B W -> B W 1")

  
    def forward(self, x, mask=None):
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

    def inverse(self, z, mask=None):
        return z.argmax(dim=-1), 0.

import torch
from survae.transforms.bijections import Bijection


# @torch.jit.script
def masked_mean(x, node_mask, dim, keepdim=False):
    return x.sum(dim=dim, keepdim=keepdim) / node_mask.sum(dim=dim, keepdim=keepdim)

# @torch.jit.script
def masked_stdev(x, node_mask, dim, keepdim=False):
    mean = masked_mean(x, node_mask, dim, keepdim=True)

    diff = (x - mean) * node_mask
    diff_2 = diff.pow(2).sum(dim=dim, keepdim=keepdim)

    diff_div_N = diff_2 / node_mask.sum(dim=dim, keepdim=keepdim)
    return torch.sqrt(diff_div_N + 1e-5)

class ActNorm(Bijection):
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims

        self.x_t = torch.nn.Parameter(torch.zeros(1, 1))
        self.x_log_s = torch.nn.Parameter(torch.zeros(1, 1))

        self.register_buffer('initialized', torch.tensor(0))

    def initialize(self, x, mask=None):
        print('initializing')
        with torch.no_grad():
            x_stdev = masked_stdev(x, mask, dim=(0, 1), keepdim=True)
            x_log_stdev = torch.log(x_stdev + 1e-8)
            self.x_log_s.data.copy_(x_log_stdev.detach())

            x_mean = masked_mean(x, mask, dim=(0, 1), keepdim=True)

            self.x_t.data.copy_(x_mean.detach())

            self.initialized.fill_(1)

    def forward(self, x, mask=None, reverse=False, logs=None):
        bs, n_nodes, dims = x.shape

        mask = mask.view(bs * n_nodes, 1)
        x = x.view(bs * n_nodes, -1).clone() * mask

        # TODO ENABLE INIT.
        if not self.initialized:
            self.initialize(x, mask=mask)


        x_log_s = self.x_log_s.expand_as(x)
        x_t = self.x_t.expand_as(x)

        x_d_ldj = -(x_log_s * mask).sum(1)
        
        d_ldj = x_d_ldj
        d_ldj = d_ldj.view(bs, n_nodes).sum(1)

        if not reverse:
            x = (x - x_t) / torch.exp(x_log_s) * mask
        else:
            x = (x * torch.exp(x_log_s) + x_t) * mask

        x = x.view(bs, n_nodes, self.n_dims)

        if not reverse:
            return x, d_ldj
        else:
            return x, -d_ldj

    def inverse(self, x, mask=None, logs=None):
        assert self.initialized
        return self(x, mask=mask, reverse=True)
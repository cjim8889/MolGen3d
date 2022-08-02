from survae.transforms.bijections import ConditionalBijection
from survae.utils import sum_except_batch
from ..utils import create_mask_equivariant
from torch.distributions import Normal
from .c_gnn import FullyConnectedGNN
import torch
from torch import nn

from collections.abc import Iterable
from survae.utils import context_size
from survae.distributions import Distribution, ConditionalDistribution
from survae.transforms import Transform, ConditionalTransform


class MaskedConditionalInverseFlow(ConditionalDistribution):
    """
    Base class for ConditionalFlow.
    Inverse flows use the forward transforms to transform noise to samples.
    These are typically useful as variational distributions.
    Here, we are not interested in the log probability of novel samples.
    However, using .sample_with_log_prob(), samples can be obtained together
    with their log probability.
    """

    def __init__(self, base_dist, transforms, context_init=None):
        super(MaskedConditionalInverseFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.context_init = context_init

    def log_prob(self, x, context):
        raise RuntimeError("ConditionalInverseFlow does not support log_prob, see ConditionalFlow instead.")

    def sample(self, context, mask=None):
        if self.context_init: context = self.context_init(context, mask=mask)
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context, mask=mask)
        else:
            z = self.base_dist.sample(context_size(context))

        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                z, _ = transform(z, context, mask=mask)
            else:
                z, _ = transform(z)
        return z

    def sample_with_log_prob(self, context, mask=None):
        if self.context_init: context = self.context_init(context, mask=mask)
        
        if isinstance(self.base_dist, ConditionalDistribution):
            z, log_prob = self.base_dist.sample_with_log_prob(context, mask=mask)
        else:
            z, log_prob = self.base_dist.sample_with_log_prob(context_size(context))

        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                z, ldj = transform(z, context, mask=mask)
            else:
                z, ldj = transform(z)
            log_prob -= ldj

        return z, log_prob

class MaskedConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(MaskedConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)

        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context, mask=None):
        dist = self.cond_dist(context)
        
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        
        if mask is not None:
            z *= mask.unsqueeze(2)
            log_prob *= mask.unsqueeze(2)

        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev


class MaskedConditionalCouplingFlow(ConditionalBijection):
    def __init__(self, ar_net, mask, split_dim=-1, no_constraint=False, last_dimension=6):
        super(MaskedConditionalCouplingFlow, self).__init__()
        
        self.ar_net = ar_net
        self.no_constraint = no_constraint
        self.register_buffer("mask", mask.unsqueeze(2))
        
        if not self.no_constraint:
            self.scaling_factor = nn.Parameter(torch.zeros(last_dimension))

        self.split_dim = split_dim

    def forward(self, x, context, mask=None):
        return self._transform(x, context, mask=mask, forward=True)

    def inverse(self, z, context, mask=None):
        return self._transform(z, context, mask=mask, forward=False)

    def _transform(self, z, context, mask=None, forward=True):
        z_masked = z * self.mask
        alpha, beta = self.ar_net(z_masked, context, mask=mask).chunk(2, dim=self.split_dim)

        if not self.no_constraint:
            # scaling factor idea inspired by UvA github to stabilise training 
            scaling_factor = self.scaling_factor.exp().view(1, 1, -1)
            alpha = torch.tanh(alpha / scaling_factor) * scaling_factor

        alpha = alpha * ~self.mask
        beta = beta * ~self.mask

        if mask is not None:
            mask = mask.unsqueeze(2)
            alpha = alpha * mask
            beta = beta * mask
        
        if forward:
            z = (z + beta) * torch.exp(alpha) # Exp to ensure invertibility
            log_det = sum_except_batch(alpha)
        else:
            z = (z * torch.exp(-alpha)) - beta
            log_det = -sum_except_batch(alpha)
        
        return z, log_det

class ConditionalARNet(nn.Module):
    def __init__(self, num_classes=6, context_dim=16, hidden_dim=64, gnn_size=2, idx=(0, 2)):
        super().__init__()
        
        self.num_classes = num_classes
        self.idx = idx
        self.net = nn.ModuleList([
            FullyConnectedGNN(
                in_dim=num_classes + context_dim,
                out_dim=num_classes + context_dim,
                m_dim=hidden_dim,
                norm_feats=True,
                soft_edges=True
            ) for _ in range(gnn_size)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(num_classes + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (self.idx[1] - self.idx[0]) * num_classes * 2),
        )
    # x: B x 9 x 6
    # context: B x 9 x 6
    def forward(self, x, context, mask=None):
        z = torch.cat((x, context), dim=-1)
        
        for gnn in self.net:
            z = gnn(z, mask=mask)
            z *= mask.unsqueeze(2)

        if mask is None:
            z = torch.mean(z, dim=1) 
        else:
            z = torch.sum(z * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)
        
        z = self.mlp(z).view(x.shape[0], self.idx[1] - self.idx[0], self.num_classes * 2)
        z = nn.functional.pad(z, (0, 0, self.idx[0], 29 - self.idx[1], 0, 0), 'constant', 0)
        
        return z

def ar_net_init(num_classes=6, context_dim=16, hidden_dim=64, gnn_size=2):
    def create(idx):
        return ConditionalARNet(num_classes=num_classes, context_dim=context_dim, hidden_dim=hidden_dim, gnn_size=gnn_size, idx=idx)

    return create

class ConditionalBlockFlow(ConditionalBijection):
    def __init__(self, ar_net_init=ar_net_init(hidden_dim=64),
            max_nodes=9,
            num_classes=6,
            partition_size=1,
            no_constraint=False,
            mask_init=create_mask_equivariant,
            split_dim=-1):

        super(ConditionalBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, partition_size):
            ar_net = ar_net_init((idx, min(idx + partition_size, max_nodes)))
            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)

            tr = MaskedConditionalCouplingFlow(ar_net, mask=mask, split_dim=split_dim, no_constraint=no_constraint, last_dimension=num_classes)
            self.transforms.append(tr)
        
       
    def forward(self, x, context, mask=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            x, ldj = transform(x, context, mask=mask)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)
        for idx in range(len(self.transforms) - 1, -1, -1):
            z, ldj = self.transforms[idx].inverse(z, context, mask=mask)
            log_prob += ldj
        
        return z, log_prob
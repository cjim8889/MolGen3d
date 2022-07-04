from .egnn import EGNN
from .residual_flows.layers import iResBlock
import torch
from torch import nn
from survae.transforms.bijections import ConditionalBijection, Bijection



class EGNN_(nn.Module):
    def __init__(self, dim, m_dim=64, num_nearest_neighbors=0, gnn_size=2) -> None:
        super().__init__()

        self.egnn = nn.ModuleList(
            [
                EGNN(
                    dim, 
                    m_dim=m_dim, 
                    num_nearest_neighbors=num_nearest_neighbors,
                    soft_edges=True,
                    update_coors=False,
                    norm_feats=True
                ) for _ in range(gnn_size)
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x

        for egnn in self.egnn:
            feats, _ = egnn(feats, feats)

        return feats

class Residual(Bijection):
    """
    Invertible residual net block, wrapper to the implementation of Chen et al.,
    see https://github.com/rtqichen/residual-flows
    """
    def __init__(self, net, n_exact_terms=2, n_samples=1, reduce_memory=True,
                 reverse=True):
        """
        Constructor
        :param net: Neural network, must be Lipschitz continuous with L < 1
        :param n_exact_terms: Number of terms always included in the power series
        :param n_samples: Number of samples used to estimate power series
        :param reduce_memory: Flag, if true Neumann series and precomputations
        for backward pass in forward pass are done
        :param reverse: Flag, if true the map f(x) = x + net(x) is applied in
        the inverse pass, otherwise it is done in forward
        """
        super().__init__()
        self.reverse = reverse
        self.iresblock = iResBlock(net, n_samples=n_samples,
                                   n_exact_terms=n_exact_terms,
                                   neumann_grad=reduce_memory,
                                   grad_in_forward=reduce_memory)

    def forward(self, z, mask=None, logs=None):
        if self.reverse:
            z, log_det = self.iresblock.inverse(z, 0)
        else:
            z, log_det = self.iresblock.forward(z, 0)
        return z, -log_det.view(-1)

    def inverse(self, z, mask=None, logs=None):
        if self.reverse:
            z, log_det = self.iresblock.forward(z, 0)
        else:
            z, log_det = self.iresblock.inverse(z, 0)
        return z, -log_det.view(-1)

class ResCoorFlow(nn.Module):
    def __init__(self, 
        hidden_dim=64, 
        gnn_size=1,
        block_size=6,
        max_nodes=29) -> None:

        super().__init__()

        self.transforms = nn.ModuleList([])

        for idx in range(block_size):
            net = EGNN_(
                3, m_dim=hidden_dim, gnn_size=gnn_size
            )

            block = Residual(
                net,
                reduce_memory=True,
                # reverse=True
            )

            self.transforms.append(block)


    def forward(self, x, context=None, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context, mask=mask)
            elif isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)

            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context=None, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):

            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context, mask=mask)
            elif isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob
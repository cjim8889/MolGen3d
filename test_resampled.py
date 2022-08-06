from lib2to3.pytree import Base
from larsflow.distributions import ResampledGaussian
from models.pos.distro_base import BaseNet
from torch import nn
import torch
max_nodes = 18
n_dim=3

net = BaseNet(
    hidden_dim=32,
    num_layers=4,
    max_nodes=18,
    n_dim=3
)

base = ResampledGaussian(
    d=max_nodes * n_dim,
    a=net,
    T=100,
    eps=0.1,
    trainable=True
)

log_p = base.log_prob(torch.randn(128, max_nodes * n_dim))

print(log_p)
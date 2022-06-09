import torch
from torch import nn
from models import CoorFlow
import matplotlib.pyplot as plt

def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def sample_with_mask(node_mask):
    base = torch.distributions.Normal(loc=0., scale=1.)
    z = base.sample(sample_shape=(node_mask.shape[0], node_mask.shape[1], 3))

    z = z * node_mask.unsqueeze(2)
    z = remove_mean_with_mask(z, node_mask)

    return z

if __name__ == "__main__":
    # net = imp.load_source('net', 'models/coordinates/coor_flow.py')

    pt = torch.load("./model_checkpoint_840.pt", map_location='cpu')

    net = CoorFlow(hidden_dim=128, gnn_size=1, block_size=1)
    net.load_state_dict(pt['model_state_dict'])



    mask = torch.ones(1, 9)
    mask[:, -1] = 0.

    mask = mask.to(torch.bool)

    z = sample_with_mask(mask)

    with torch.no_grad():
        x, _ = net.inverse(z, mask=mask)

    # print(z, x)
    x = x[0, :-1]
    xs = x[:, 0]
    ys = x[:, 1]
    zs = x[:, 2]

    print(xs, ys, zs)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    # print(net)
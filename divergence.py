import enum
import torch
from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
import matplotlib
from models.pos.flow import TransformerCoorFlowV2
from models.pos.distro_base import BaseNet
from larsflow.distributions import ResampledGaussian

import torch
import numpy as np
from einops import rearrange

# matplotlib.use("MacOSX")

@torch.jit.script
def remove_mean_with_constraint(x, size_constraint):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x

def normalize_histogram(hist):
    hist = np.array(hist)
    prob = hist / np.sum(hist)
    return prob

def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    dist = dist.flatten()
    return dist

def kl_divergence(p1, p2):
    return np.sum(p1*np.log(p1 / p2))



def js_divergence(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10

    M = (p1 + p2)/2
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2
    return js


class Histogram_cont:
    def __init__(self, num_bins=100, range=[0., 13.], name='histogram', ignore_zeros=False):
        self.name = name
        self.bins = [0] * num_bins
        self.range = range
        self.ignore_zeros = ignore_zeros

    def add(self, elements):
        for e in elements:
            if not self.ignore_zeros or e > 1e-8:
                i = int(float(e) / self.range[1] * len(self.bins))
                i = min(i, len(self.bins) - 1)
                self.bins[i] += 1

    def plot(self, save_path=None):
        width = (self.range[1] - self.range[0])/len(self.bins) # the width of the bars
        fig, ax = plt.subplots()

        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1] + width / 2
        ax.bar(x, self.bins, width)
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    def plot_both(self, hist_b, save_path=None, wandb=None):
        ## TO DO: Check if the relation of bins and linspace is correct
        hist_a = normalize_histogram(self.bins)
        hist_b = normalize_histogram(hist_b)

        #width = (self.range[1] - self.range[0]) / len(self.bins)  # the width of the bars
        fig, ax = plt.subplots()
        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
        ax.step(x, hist_b)
        ax.step(x, hist_a)
        ax.legend(['QM9-19', 'Learned'])
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
            if wandb is not None:
                if wandb is not None:
                    # Log image(s)
                    im = plt.imread(save_path)
                    wandb.log({save_path: [wandb.Image(im, caption=save_path)]})
        else:
            plt.show()
        plt.close()

if __name__ == "__main__":
    # train_loader, test_loader = get_datasets(type="mqm9")
    base_normal = torch.distributions.Normal(loc=0., scale=1.)

    batch_size = 2000
    mol_size = 18
    resampled = False


    coor_net = TransformerCoorFlowV2(
        hidden_dim=128,
        num_layers_transformer=8,
        block_size=6,
        max_nodes=18,
        conv1x1=True,
        conv1x1_node_wise=True,
        batch_norm=False,
        act_norm=True,
        partition_size=(1,6),
        squeeze=True,
        squeeze_step=2
    )
    
    states = torch.load("outputs/model_checkpoint_2o3qumlf_2400.pt", map_location="cpu")
    
    coor_net.load_state_dict(
        states['model_state_dict']
    )

    if resampled:
        net = BaseNet(
            hidden_dim=128,
            num_layers=8,
            max_nodes=18,
            n_dim=3,
        )

        base = ResampledGaussian(
            d=18 * 3,
            a=net,
            T=100,
            eps=0.1,
            trainable=True
        )

        base.load_state_dict(
            states['base']
        )
        base.eval()

    print("Loaded TransformerCoorFlow model...")

    print("Sampling...")

    if resampled:
        z, _ = base.forward(num_samples=batch_size)
        z = rearrange(z, "b (d n) -> b d n", d=3)
    else:
        z = torch.randn(batch_size, mol_size, 3,)
        # z = remove_mean_with_constraint(z, mol_size)
        z = rearrange(z, "b d n -> b n d")
    
    print(z.shape)
    with torch.no_grad():
        pos, _ = coor_net.inverse(
            z,
        )

    print("Sampled Positions...")
    pos = rearrange(pos, "b d n -> b n d")
    dist = coord2distances(pos)
    print(dist[:20])

    hist = Histogram_cont(name='Histogram for Relative Distance Between Nodes', ignore_zeros=True)
    hist.add(list(dist.numpy()))

    train_loader, test_loader = get_datasets(type="mqm9", batch_size=100, size_constraint=18)
    hist_qm9 = Histogram_cont(name='histogram', ignore_zeros=True)

    for idx, batch in enumerate(train_loader):
        if idx >= 20:
            break
        qm9_pos = batch.pos
        qm9_dist = coord2distances(qm9_pos)
        hist_qm9.add(list(qm9_dist.numpy()))
   

    hist.plot_both(hist_qm9.bins, save_path="histogram.pdf", wandb=None)

    js_div = js_divergence(normalize_histogram(hist.bins), normalize_histogram(hist_qm9.bins))

    print("JS Divergence: ", js_div)
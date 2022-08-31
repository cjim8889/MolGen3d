from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from models.argmax.atom import AtomFlow
from models import CoorFlow
import torch
import matplotlib
import numpy as np
from xyz2mol.xyz2mol import xyz2mol

atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_decoder_int = [1, 6, 7, 8, 9]
# matplotlib.use("MacOSX")

class Histogram_discrete:
    def __init__(self, name='histogram'):
        self.name = name
        self.bins = {}

    def add(self, elements):
        for e in elements:
            if e in self.bins:
                self.bins[e] += 1
            else:
                self.bins[e] = 1

    def normalize(self):
        total = 0.
        for key in self.bins:
            total += self.bins[key]
        for key in self.bins:
            self.bins[key] = self.bins[key] / total

    def plot(self, hist_a, save_path=None):
        width = 1  # the width of the bars
        fig, axes = plt.subplots(1, 2)

        x, y = [], []
        for key in self.bins:
            x.append(key)
            y.append(self.bins[key])

        axes[0].bar(x, y, width)

        x, y = [], []
        for key in hist_a.bins:
            x.append(key)
            y.append(hist_a.bins[key])

        axes[1].bar(x, y, width, alpha=0.5)

        axes[0].legend(['QM9'])
        axes[1].legend(['Learned'])

        fig.suptitle(self.name)
        # plt.title(self.name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

if __name__ == "__main__":
    net = AtomFlow(
        hidden_dim=32,
        block_size=6,
        encoder_size=4,
        gnn_size=2,
        num_classes=5,
        stochastic_permute=False
    )

    net.load_state_dict(
        torch.load("outputs/model_checkpoint_3pchowk4_200.pt", map_location="cpu")['model_state_dict']
    )

    print("Loading model...")

    train_loader, test_loader = get_datasets(type="mqm9", batch_size=128)

    print("Loading data...")
    total_count = 0
    validity = 0
    valid_smiles = []

    hist_qm9 = Histogram_discrete("Histogram for Types of Sampled Nodes")
    hist_learned = Histogram_discrete("Histogram for Types of Learned Nodes")
    
    # r = torch.randint(0, 5, (100,)).numpy()
    # hist.add(r)
    # hist.normalize()

    # hist.plot()
    for idx, batch_data in enumerate(train_loader):

        if idx > 10:
            break
        pos = batch_data.pos
        mask = batch_data.mask

        x = batch_data.x.to(torch.long).squeeze(2).numpy()

        with torch.no_grad():
            atoms_types, _ = net.inverse(
                torch.randn(pos.shape[0], 29, 5),
                pos,
                mask = mask
            )

        atoms_types = atoms_types.long().numpy()
        pos = pos.numpy()

        for idx in range(pos.shape[0]):
            size = mask[idx].to(torch.long).sum()
            atom_ty =[atom_decoder[i] for i in atoms_types[idx, :size]]
            hist_learned.add(atom_ty)

            atom_ty_x = [atom_decoder[i] for i in x[idx, :size]]
            hist_qm9.add(atom_ty_x)
    
    hist_qm9.normalize()
    hist_learned.normalize()

    hist_qm9.plot(hist_learned, save_path="histogram_discrete.pdf")

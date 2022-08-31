from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from models.argmax.atom import AtomFlow
from models import CoorFlow
import torch
import numpy as np
from xyz2mol.xyz2mol import xyz2mol

atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_decoder_int = [1, 6, 7, 8, 9]


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

    def plot(self, save_path=None):
        width = 1  # the width of the bars
        fig, ax = plt.subplots()
        x, y = [], []
        for key in self.bins:
            x.append(key)
            y.append(self.bins[key])

        ax.bar(x, y, width)
        plt.title(self.name)
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
        torch.load("outputs/model_checkpoint_3pchowk4_65.pt", map_location="cpu")['model_state_dict']
    )

    print("Loading model...")

    train_loader, test_loader = get_datasets(type="mqm9", batch_size=128)

    print("Loading data...")
    total_count = 0
    validity = 0
    valid_smiles = []

    hist = Histogram_discrete()

    for idx, batch_data in enumerate(train_loader):

        pos = batch_data.pos
        mask = batch_data.mask

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

            pos_t = pos[idx, :size].tolist()

            total_count += 1

            try:
                # print(pos_t, atom_ty)
                mols = xyz2mol(
                    atom_ty,
                    pos_t,
                    use_huckel=True,
                )


                for mol in mols:
                    smiles = Chem.MolToSmiles(mol)

                    if "." in smiles:
                        continue
                    else:
                        validity += 1
                        valid_smiles.append(smiles)
                        print(smiles)
                        break
            except:
                pass
        
        print(validity * 1.0 / total_count)

    # pprint(valid_smiles)
        


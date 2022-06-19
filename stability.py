from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from models.argmax import AtomFlow
from models import CoorFlow
import torch
import numpy as np
from xyz2mol.xyz2mol import xyz2mol

atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
atom_decoder_int = [0, 1, 6, 7, 8, 9]

def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def get_mol(atom_types, pos):
    pass

# 0.029
# 0.016

if __name__ == "__main__":
    # train_loader, test_loader = get_datasets(type="mqm9")
    base = torch.distributions.Normal(loc=0., scale=1.)
    # batch = next(iter(train_loader))

    # pos = batch.pos
    # mask = batch.mask
    batch_size = 1000

    coor_net = CoorFlow(hidden_dim=64, gnn_size=1, block_size=8)
    coor_net.load_state_dict(
        torch.load("model_checkpoint_27pnk9mt_550.pt", map_location="cpu")['model_state_dict']
    )

    z = base.sample(sample_shape=(batch_size, 29, 3))
    mask = torch.ones(batch_size, 29).to(torch.bool)
    mask_size = torch.randint(5, 29, (batch_size,))
    
    for idx in range(batch_size):
        mask[idx, mask_size[idx]:] = False

    
    z = z * mask.unsqueeze(2)
    z = remove_mean_with_mask(z, node_mask=mask)

    with torch.no_grad():
        pos, _ = coor_net.inverse(z, mask=mask)

    # # print(pos[0])
    net = AtomFlow(
        hidden_dim=64,
        block_size=3
    )

    net.load_state_dict(
        torch.load("model_checkpoint_2gq9kaav_490.pt", map_location="cpu")['model_state_dict']
    )

    with torch.no_grad():
        atoms_types, _ =net.inverse(
            base.sample(sample_shape=(pos.shape[0], 29, 6)),
            pos,
            mask = mask
        )

    valid = 0
    atoms_types = atoms_types.long().numpy()
    pos = pos.numpy()

    valid_smiles =[]
    # print(atoms_types[0])
    for idx in range(atoms_types.shape[0]):
        size = mask[idx].to(torch.long).sum()
        atom_decoder_int = [0, 1, 6, 7, 8, 9]
        atom_ty =[atom_decoder_int[i] for i in atoms_types[idx, :size]]

        pos_t = pos[idx, :size].tolist()

        if 0 in atom_ty or len(atom_ty) == 0:
            continue

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
                    valid += 1
                    valid_smiles.append(smiles)
                    break
        except:
            pass

    pprint(valid_smiles)
    print(valid * 1.0 / batch_size)
    
    # # # print(atom_types, pos[0:1], x[0: 1])

    
    # # # print(pos[0])
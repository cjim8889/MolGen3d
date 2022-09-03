from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from models.pos.flow import ConditionalTransformerCoorFlow

import torch
import numpy as np
from einops import rearrange
from xyz2mol.xyz2mol import xyz2mol

atom_decoder = ['H', 'C', 'N', 'O', 'F']


@torch.jit.script
def remove_mean_with_constraint(x, size_constraint):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x

if __name__ == "__main__":
    

    # pos = batch.pos
    # mask = batch.mask

    batch_size = 200

    train_loader, test_loader = get_datasets(type="mqm9", size_constraint=18, batch_size=batch_size)
    base_normal = torch.distributions.Normal(loc=0., scale=1.)
    batch = next(iter(train_loader))

    coor_net = ConditionalTransformerCoorFlow(
        hidden_dim=128,
        num_layers_transformer=8,
        block_size=8,
        max_nodes=18,
        conv1x1=True,
        conv1x1_node_wise=True,
        act_norm=True,
        partition_size=(1,6),
        squeeze=False,
        squeeze_step=2
    )

    states = torch.load("outputs/model_checkpoint_2h3lcmp9_1300.pt", map_location="cpu")
    
    coor_net.load_state_dict(
        states['model_state_dict']
    )

    print("Loaded TransformerCoorFlow model...")

    print("Sampling...")

    mol_size = 18

    z = torch.randn(batch_size, mol_size, 3,)
    z = remove_mean_with_constraint(z, mol_size)
    z = rearrange(z, "b d n -> b n d")

    x = batch.x.squeeze(2).long()
    print(z.shape)
    with torch.no_grad():
        pos, _ = coor_net.inverse(
            z, x
        )

    print("Sampled Positions...")
    print(pos.shape)

    valid = 0

    atoms_types_n = x.numpy()
    pos_n = pos.numpy()

    valid_smiles =[]
    valid_mols = []
    valid_idx = []

    invalid_idx = []

    for idx in range(atoms_types_n.shape[0]):
        size = mol_size
        atom_decoder_int = [1, 6, 7, 8, 9]
        atom_ty =[atom_decoder_int[i] for i in atoms_types_n[idx, :size]]

        pos_t = pos_n[idx, :size].tolist()


        try:
            mols = xyz2mol(
                atom_ty,
                pos_t,
                use_huckel=True,
            )


            for mol in mols:
                smiles = Chem.MolToSmiles(mol)

                valid += 1
                valid_idx.append(idx)
                valid_smiles.append(smiles)
                valid_mols.append(mol)
                break
        except:
            invalid_idx.append(idx)


    pprint(valid_smiles)
    print(valid * 1.0 / batch_size)
    

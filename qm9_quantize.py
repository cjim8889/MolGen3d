from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import numpy as np
from xyz2mol.xyz2mol import xyz2mol
atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_decoder_int = [1, 6, 7, 8, 9]
from xyz2mol.xyz2mol import xyz2mol

if __name__ == "__main__":
    train_loader, test_loader = get_datasets(type="mqm9", batch_size=128, num_workers=4)

    total_count = 0
    validity = 0
    valid_smiles = []

    for batch_data in train_loader:

        pos = batch_data.pos
        mask = batch_data.mask
        atoms_types = batch_data.x.long()

        atoms_types = atoms_types.squeeze(2).numpy()

        pos = (torch.floor(pos * 100.0) + torch.randint_like(pos, 0, 5)) / 100.0
        pos = pos.numpy()

        for idx in range(pos.shape[0]):
            size = mask[idx].to(torch.long).sum()
            atom_decoder_int = [1, 6, 7, 8, 9]
            atom_ty =[atom_decoder_int[i] for i in atoms_types[idx, :size]]

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
                        break
            except:
                pass
        
        print(validity * 1.0 / total_count)

    # pprint(valid_smiles)
        


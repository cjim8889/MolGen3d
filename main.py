from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from xyz2mol.xyz2mol import xyz2mol

bond_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}

def get_mol(atom_map, adj_dense, verbose=True):
    mol = Chem.RWMol()

    end_index = -1
    for atom_index in range(atom_map.shape[0]):
        if atom_map[atom_index, 0] == 0:
            end_index = atom_index
            break

        mol.AddAtom(Chem.Atom(atom_decoder[atom_map[atom_index, 0]]))
    
    if end_index == -1:
        end_index = 9


    for i in range(end_index):
        for j in range(i + 1, end_index):
            if adj_dense[i, j] == 4:
                continue

            
            bond_rdkit = bond_dict[int(adj_dense[i, j])]
            mol.AddBond(i, j, bond_rdkit)
    
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)

        if verbose:
            print(f"Chemically Correct Molecule: {smiles}")

        return mol, smiles
    except:
        return mol, None

if __name__ == "__main__":
    train_loader, test_loader = get_datasets(type="mqm9")

    batch = next(iter(train_loader))

    
    mol_idx = 0

    pos = batch.pos[mol_idx]
    x = batch.x[mol_idx, :, 0].long().numpy()
    smiles = batch.smiles[mol_idx]
    mask = batch.mask[mol_idx]

    size = mask.to(torch.long).sum()

    atom_decoder_int = [0, 1, 6, 7, 8, 9]

    atom_ty =[atom_decoder_int[i] for i in x[:size]]
    pos = pos.numpy().tolist()


    mols = xyz2mol(
        atom_ty,
        pos,
        use_huckel=True,
    )

    print(smiles)

    smiles_t = Chem.MolToSmiles(mols[0])
    print(smiles_t)
    


from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

bond_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
atom_decoder = ['B', 'H', 'C', 'N', 'O', 'F']
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

    mol = batch.pos[0]

    # print(mol[0])
    xs = mol[:, 0]
    ys = mol[:, 1]
    zs = mol[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    # adj_dense = batch.orig_adj[0].argmax(dim=-1)
    # x = batch.x[0].long()
    # mol, smile = get_mol(x, adj_dense, verbose=False)
    # print(smile)

    # plt.show()

    


import numpy as np
import torch
from models.argmax import AtomFlow
from models import CoorFlow
atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
          'C': {'H': 109, 'C': 154 , 'N': 147, 'O': 143, 'F': 135},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142}}

bonds2 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 134, 'N': 129, 'O': 120, 'F': -1000},
          'N': {'H': -1000, 'C': 129, 'N': 125, 'O': 121, 'F': -1000},
          'O': {'H': -1000, 'C': 120, 'N': 121, 'O': 121, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}

bonds3 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 120, 'N': 116, 'O': 113, 'F': -1000},
          'N': {'H': -1000, 'C': 116, 'N': 110, 'O': -1000, 'F': -1000},
          'O': {'H': -1000, 'C': 113, 'N': -1000, 'O': -1000, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0

def check_stability(positions, atom_type, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[
                atom_type[j]]
            order = get_bond_order(atom1, atom2, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        is_stable = allowed_bonds[atom_decoder[atom_type_i]] == nr_bonds_i
        if is_stable == False and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

if __name__ == "__main__":
    # train_loader, test_loader = get_datasets(type="mqm9")
    base = torch.distributions.Normal(loc=0., scale=1.)
    # batch = next(iter(train_loader))

    # pos = batch.pos
    # mask = batch.mask
    batch_size = 200

    coor_net = CoorFlow(hidden_dim=64, gnn_size=1, block_size=4)
    coor_net.load_state_dict(
        torch.load("model_checkpoint_1eac4ec9_590.pt", map_location="cpu")['model_state_dict']
    )

    z = base.sample(sample_shape=(batch_size, 29, 3))
    mask = torch.ones(batch_size, 29).to(torch.bool)
    mask_size = torch.randint(3, 29, (batch_size,))
    
    for idx in range(batch_size):
        mask[idx, mask_size[idx]:] = False


    
    z = z * mask.unsqueeze(2)
    z = remove_mean_with_mask(z, node_mask=mask)

    with torch.no_grad():
        pos, _ = coor_net.inverse(z, mask=mask)
    # train_loader, _ = get_datasets(type="mqm9", batch_size=batch_size)

    # batch = next(iter(train_loader))

    # pos = batch.pos
    # mask = batch.mask

    # # print(pos[0])
    net = AtomFlow(
        hidden_dim=128,
        block_size=10,
        encoder_size=5
    )

    net.load_state_dict(
        torch.load("model_checkpoint_1642pd3z_80.pt", map_location="cpu")['model_state_dict']
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
    valid_mols = []
    # print(atoms_types[0])
    for idx in range(atoms_types.shape[0]):
        size = mask[idx].to(torch.long).sum()
        
        atom_ty =atoms_types[idx, :size]
        pos_t = pos[idx, :size]

        if 0 in atom_ty or len(atom_ty) == 0:
            print("skipped")
            continue
        
        validity, _, _ = check_stability(pos_t, atom_ty, debug=False)
        if validity:
            valid += 1

        
    
    # plot = Draw.MolsToGridImage(valid_mols, molsPerRow=4, subImgSize=(500, 500), legends=valid_smiles)
    # number = np.random.randint(0, 10000)
    # plot.save(f"local_interpolcation_{number}.png")

    print(valid * 1.0 / batch_size)
    
    # # # print(atom_types, pos[0:1], x[0: 1])

    
    # # # print(pos[0])
from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from models.argmax.atom import AtomFlow
from models.pos.flow import TransformerCoorFlow, TransformerCoorFlowV2
from models.pos.distro_base import BaseNet
from larsflow.distributions import ResampledGaussian
from models.classifier import PosClassifier
from utils.visualise import plot_data3d

import torch
import numpy as np
from einops import rearrange
from xyz2mol.xyz2mol import xyz2mol

# atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
# atom_decoder_int = [0, 1, 6, 7, 8, 9]

atom_decoder = ['H', 'C', 'N', 'O', 'F']
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



@torch.jit.script
def remove_mean_with_constraint(x, size_constraint):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x
# 0.029
# 0.016
# 0.018

# 0.026
# 0.026
# 0.024
# 0.026
# 0.018
# 0.024
# 0.032
# 0.034
# 0.036
if __name__ == "__main__":
    # train_loader, test_loader = get_datasets(type="mqm9")
    base_normal = torch.distributions.Normal(loc=0., scale=1.)
    # batch = next(iter(train_loader))

    # pos = batch.pos
    # mask = batch.mask

    resampled = True
    batch_size = 200

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

    states = torch.load("outputs/model_checkpoint_1mghyk3o_3800.pt", map_location="cpu")
    
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

    mol_size = 18

    if resampled:
        z, _ = base.forward(num_samples=batch_size)
        z = rearrange(z, "b (d n) -> b d n", d=3)
    else:
        z = torch.randn(batch_size, mol_size, 3,)
        z = remove_mean_with_constraint(z, mol_size)
        z = rearrange(z, "b d n -> b n d")

    # z = remove_mean_with_constraint(z, mol_size)

    print(z.shape)
    with torch.no_grad():
        pos, _ = coor_net.inverse(
            z,
        )

    print("Sampled Positions...")
    print(pos.shape)

    classifier = PosClassifier(feats_dim=64, hidden_dim=256, gnn_size=5)
    classifier.load_state_dict(torch.load("classifier.pt", map_location="cpu")['model_state_dict'])



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

    print("Loaded AtomFlow model...")


    zeros = torch.zeros(batch_size, 3, 29 - mol_size)
    pos = torch.cat([pos, zeros], dim=2)
    pos = rearrange(pos, "b d n -> b n d")
    mask = torch.ones(batch_size, 29, dtype=torch.bool)
    mask[:, mol_size:] = False

    with torch.no_grad():
        output = torch.sigmoid(classifier(pos, mask=mask)).squeeze()
        print(output.sum())

    
    with torch.no_grad():
        atoms_types, _ =net.inverse(
            base_normal.sample(sample_shape=(pos.shape[0], 29, 5)) * mask.unsqueeze(2),
            pos,
            mask = mask
        )

    print("Sampled Atom Types...")
    valid = 0
    # atoms_types = atoms_types.long().numpy()
    # pos = pos.numpy()

    # valid_smiles =[]
    # valid_mols = []
    # # print(atoms_types[0])
    # for idx in range(atoms_types.shape[0]):
    #     size = mask[idx].to(torch.long).sum()
        
    #     atom_ty =atoms_types[idx, :size]
    #     pos_t = pos[idx, :size]

    #     # if 0 in atom_ty or len(atom_ty) == 0:
    #         # print("skipped")
    #         # continue
        
    #     validity, _, _ = check_stability(pos_t, atom_ty, debug=True)
    #     if validity:
    #         print("1")
    #         valid += 1
    
    # print(valid)
    atoms_types_n = atoms_types.long().numpy()
    pos_n = pos.numpy()

    valid_smiles =[]
    valid_mols = []
    valid_idx = []
    # print(atoms_types[0])

    invalid_idx = []

    for idx in range(atoms_types.shape[0]):
        size = mask[idx].to(torch.long).sum()
        atom_decoder_int = [1, 6, 7, 8, 9]
        atom_ty =[atom_decoder_int[i] for i in atoms_types_n[idx, :size]]

        pos_t = pos_n[idx, :size].tolist()


        try:
            # print(pos_t, atom_ty)
            mols = xyz2mol(
                atom_ty,
                pos_t,
                use_huckel=True,
            )


            for mol in mols:
                smiles = Chem.MolToSmiles(mol)

                # if "." in smiles:

                #     aty = atoms_types_n[idx, :size].reshape(-1)
                #     p = pos_n[idx].reshape(-1, 3)
                    
                #     plot_data3d(
                #         positions=p,
                #         atom_type=aty,
                #         spheres_3d=True,
                #     )
                #     invalid_idx.append(idx)
                #     continue
                # else:
                valid += 1
                valid_idx.append(idx)
                valid_smiles.append(smiles)
                valid_mols.append(mol)
                break
        except:
            # aty = atoms_types[idx, :size].view(-1).numpy()
            # p = pos[idx, :size].view(-1, 3).numpy()
            
            # plot_data3d(
            #     positions=p,
            #     atom_type=aty,
            #     spheres_3d=True,
            #     save_path="outputs/invalid_" + str(idx) + ".png"
            # )

            invalid_idx.append(idx)


    pprint(valid_smiles)
    print(valid * 1.0 / batch_size)
    

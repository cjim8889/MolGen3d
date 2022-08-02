from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from models.argmax.atom import AtomFlow
from models.pos.flow import TransformerCoorFlow
import torch
import numpy as np
from einops import rearrange
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
    base = torch.distributions.Normal(loc=0., scale=1.)
    # batch = next(iter(train_loader))

    # pos = batch.pos
    # mask = batch.mask
    batch_size = 100

    coor_net = TransformerCoorFlow(
        hidden_dim=32,
        num_layers_transformer=4,
        block_size=6
    )
    
    coor_net.load_state_dict(
        torch.load("outputs/model_checkpoint_2yd53373_30.pt", map_location="cpu")['model_state_dict']
    )

    print("Loaded TransformerCoorFlow model...")

    print("Sampling...")
    z = base.sample(sample_shape=(batch_size, 29, 3))
    mask = torch.ones(batch_size, 29).to(torch.bool)
    mask_size = torch.randint(3, 29, (batch_size,))
    
    for idx in range(batch_size):
        mask[idx, mask_size[idx]:] = False

    z = z * mask.unsqueeze(2)
    z = remove_mean_with_mask(z, node_mask=mask)

    with torch.no_grad():
        pos, _ = coor_net.inverse(
            rearrange(z, "b n d -> b d n"),
            mask=mask.unsqueeze(1)
        )

    print("Sampled Positions...")
    print(pos.shape)

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


    pos = rearrange(pos, "b d n -> b n d") * mask.squeeze(1).unsqueeze(2)

    with torch.no_grad():
        atoms_types, _ =net.inverse(
            base.sample(sample_shape=(pos.shape[0], 29, 5)),
            pos,
            mask = mask.squeeze(1)
        )

    print("Sampled Atom Types...")
    valid = 0
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

                if "." in smiles:
                    invalid_idx.append(idx)
                    continue
                else:
                    valid += 1
                    valid_idx.append(idx)
                    valid_smiles.append(smiles)
                    valid_mols.append(mol)
                    break
        except:
            invalid_idx.append(idx)

    # print(invalid_idx)
    # print(pos[invalid_idx].shape, pos[invalid_idx])
    
    # plot = Draw.MolsToGridImage(valid_mols, molsPerRow=4, subImgSize=(500, 500), legends=valid_smiles)
    # number = np.random.randint(0, 10000)
    # plot.save(f"local_interpolcation_{number}.png")

    pprint(valid_smiles)
    print(valid * 1.0 / batch_size)
    

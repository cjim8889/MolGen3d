from utils import get_datasets
import matplotlib.pyplot as plt
import matplotlib
from rdkit import Chem
import torch
from models import CoorFlow
import matplotlib.animation as animation
from pprint import pprint
import numpy as np
import copy

bond_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}


if __name__ == "__main__":
    train_loader, test_loader = get_datasets(type="mqm9")

    batch = next(iter(train_loader))

    coor_net = CoorFlow(hidden_dim=128, gnn_size=1, block_size=2, max_nodes=9)
    coor_net.load_state_dict(
        torch.load("model_checkpoint_950.pt", map_location="cpu")['model_state_dict']
    )

    # print(batch.pos, batch.mask)
    mol_idx = 0

    pos = batch.pos
    mask = batch.mask

    logs = []

    with torch.no_grad():
        z, _ = coor_net(pos, mask=mask, logs=logs)

    print(len(logs))


    idx = 0
    steps = len(logs)

    pos = batch.pos[idx].numpy()
    print(pos)
    # matplotlib.use("TkAgg")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    points = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "o")

    def update(num, data, points):
        t = data[num].numpy()

        points[0].set_data(t[:, 0:2].T)
        points[0].set_3d_properties(t[:, 2])

        ax.set_title(f'Step {num}')
        return points

    # data = [np.random.rand(9, 3) for i in range(50)]
    ax.set(xlim3d=(-5, 5), xlabel='X')
    ax.set(ylim3d=(-5, 5), ylabel='Y')
    ax.set(zlim3d=(-5, 5), zlabel='Z')

    data = [k[idx] for k in logs]
    ani = animation.FuncAnimation(
        fig, update, steps, fargs=(data, points), interval=200)

    ani.save("./animation.gif", writer='imagemagick', fps=12)
    plt.show()
    # smiles_t = Chem.MolToSmiles(mols[0])
    # print(smiles_t)
    


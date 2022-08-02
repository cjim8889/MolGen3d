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


if __name__ == "__main__":
    train_loader, test_loader = get_datasets(type="mqm9", batch_size=128, num_workers=4, size_constraint=18)

    dict = dict()
    for batch_data in train_loader:
        size = batch_data.mask.sum(dim=-1)
        
        for s in size:
            tmp = s.item()
            if tmp not in dict:
                dict[tmp] = 1
            else:
                dict[tmp] += 1

        pprint(dict)
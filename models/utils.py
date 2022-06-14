import torch
import numpy as np
'''
Verticle mask to make to model equivariant
'''
def create_mask_equivariant(idx, max_size, invert=False):

    mask = torch.zeros(max_size, dtype=torch.float32)

    mask[idx] = 1
    mask = mask.view(1, max_size)

    if not invert:
        mask = 1 - mask

    return mask.to(torch.bool)

def create_mask_ar(idx, max_shape, invert=False):
    mask = torch.zeros(max_shape, dtype=torch.float32)

    
    row_idx =  idx // max_shape[1]
    col_idx =  idx % max_shape[1]

    mask[row_idx, col_idx] = 1
    mask = mask.view(1, max_shape[0], max_shape[1])

    if not invert:
        mask = 1 - mask

    return mask
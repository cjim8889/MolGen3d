import torch


def create_mask_equivariant(idx, max_size, invert=False):

    mask = torch.zeros(max_size, dtype=torch.bool)

    mask[idx] = True
    mask = mask.view(1, max_size)

    if not invert:
        mask = ~mask

    return mask

def create_mask_ar(idx, max_shape, invert=False):
    mask = torch.zeros(max_shape, dtype=torch.float32)

    
    row_idx =  idx // max_shape[1]
    col_idx =  idx % max_shape[1]

    mask[row_idx, col_idx] = 1
    mask = mask.view(1, max_shape[0], max_shape[1])

    if not invert:
        mask = 1 - mask

    return mask
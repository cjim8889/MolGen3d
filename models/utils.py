import torch

'''
Verticle mask to make to model equivariant
'''
def create_mask_equivariant(idx, max_size, invert=False):

    mask = torch.zeros(max_size, dtype=torch.float32)

    mask[idx] = 1
    mask = mask.view(1, 1, max_size)

    if not invert:
        mask = 1 - mask

    return mask
from survae.transforms.surjections import Surjection
import torch

class NodeWiseStochasticPermutation(Surjection):
    stochastic_forward = True

    def __init__(self):
        super(NodeWiseStochasticPermutation, self).__init__()

    def permute_with_mask_(self, x, pos, mask=None):
        size = mask.sum(dim=(1))

        for idx in range(x.shape[0]):
            perm = torch.randperm(size[idx], device=x.device)

            z = x[idx].index_select(0, perm)
            x[idx, :size[idx], :] = z

            z_pos = pos[idx].index_select(0, perm)
            pos[idx, :size[idx], :] = z_pos
    
    def forward(self, x, pos, mask=None, logs=None):
        # Change both x and pos in-place
        x.unsqueeze_(2)

        self.permute_with_mask_(x, pos, mask=mask)
        log_det = torch.zeros(x.shape[0], device=x.device)

        x.squeeze_(2)
        
        return x, log_det

    def inverse(self, z, pos, mask=None):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det
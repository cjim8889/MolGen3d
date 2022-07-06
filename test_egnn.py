from venv import create
from models.egnn import EGNN, ResCoorFlow
from models.egnn.residual_flows.layers import iResBlock
from models.actnorm import ActNorm
from models.utils import create_mask_col
from models.block import ARNet, CouplingBlockFlow
from models.coordinates import CoorFlow
import torch
from torch import nn
class EGNN_(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.egnn = EGNN(
            3, 
            m_dim=64, 
            num_nearest_neighbors=0,
            soft_edges=True,
            update_coors=False,
            norm_feats=True
        )
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        feats, _ = self.egnn(x, x, mask=mask)
        return feats


if __name__ == "__main__":

    # net = ResCoorFlow(
    #     hidden_dim=16,
    #     block_size=1
    # )

    # feats = torch.randn(10, 10, 3)
    # mask = torch.ones(10, 10, dtype=torch.bool)

    # mask[:, -2:] = False

    # net = ActNorm(3)
    # net = ARNet(hidden_dim=16, gnn_size=1, idx=0, activation='LipSwish')
    # mask = create_mask_col(0, 6)
    # print(mask, mask.shape)
    # net = CouplingBlockFlow(last_dimension=6)
    net = CoorFlow(hidden_dim=64, gnn_size=1, block_size=1)
    val = torch.randn(1, 29, 6)
    mask = torch.ones(1, 29, dtype=torch.bool)
    

    out, _ = net(val, mask=mask)
    print(out.shape)

    # k, ldj = net(feats, mask=mask)
    # print(feats[0], k[0])
    # net.initialize(feats.view(feats.shape[0] * feats.shape[1], -1), mask.view(feats.shape[0] * feats.shape[1], -1))

    # mask = torch.ones(1, 10, dtype=torch.bool)
    # mask[:, -2:] = False

    # feats = feats * mask.unsqueeze(2)
    # feats += torch.ones_like(feats) * ~mask.unsqueeze(2)
    # feats = net(feats)
    # feats = feats.detach()


    # z, logp = net(feats)
    # print(z, logp)
    # x, logp = net.inverse(z)
    # 
    # print(z)
    # print(feats)
    # print(x)
    # x = x * mask.unsqueeze(2)
    # print(x, feats)
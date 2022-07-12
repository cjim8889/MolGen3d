from models.egnn import ModifiedPosEGNN, ResCoorFlow
from models.coordinates import CoorFlow
from models.block import ARNet, CouplingBlockFlow
from models.egnn import ModifiedPosEGNN
from models.egnn.residual_flows.layers import iResBlock
from models.actnorm import ActNorm
import torch
from torch import nn
class EGNN_(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.egnn = ModifiedPosEGNN(
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

    feats = torch.randn(1, 29, 3)
    mask = torch.ones(1, 29, dtype=torch.bool)

    mask[:, -2:] = False

    # net = ModifiedPosEGNN(in_dim=3, out_dim=6, norm_coors=True, fourier_features=2, soft_edges=True, dropout=0.1)
    # net = ARNet(hidden_dim=64, gnn_size=1, idx=(0, 2), activation='LipSwish')
    net = CoorFlow(hidden_dim=16, gnn_size=2, block_size=2, activation='SiLU', act_norm=False)
    # net = CouplingBlockFlow(act_norm=False, partition_size=4)
    print(net)
    print(feats)
    feats, _ = net(feats, mask=mask)
    print(feats.shape)

    print(feats)
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
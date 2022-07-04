from models.egnn import EGNN, ResCoorFlow
from models.egnn.residual_flows.layers import iResBlock
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

    net = ResCoorFlow(
        hidden_dim=16,
        block_size=1
    )

    feats = torch.randn(1, 10, 3)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, -2:] = False

    feats = feats * mask.unsqueeze(2)
    # feats += torch.ones_like(feats) * ~mask.unsqueeze(2)
    # feats = net(feats)
    feats = feats.detach()


    z, logp = net(feats)
    # print(z, logp)
    x, logp = net.inverse(z)
    
    print(z)
    print(feats)
    print(x)
    # x = x * mask.unsqueeze(2)
    # print(x, feats)
import torch
from torch import nn
from egnn_pytorch import EGNN

class PosClassifier(nn.Module):
    def __init__(self, feats_dim=16, hidden_dim=64, gnn_size=3) -> None:
        super().__init__()

        self.feats_dim = feats_dim
        self.net = nn.ModuleList(
            [
                EGNN(
                    feats_dim,
                    m_dim=hidden_dim,
                    num_nearest_neighbors=6,
                    norm_coors=True,
                    soft_edges=True,
                    coor_weights_clamp_value=2.,
                    fourier_features=2
                )
            ]
        )

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(hidden_dim * 2),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

    def forward(self, pos, mask=None):
        feats = torch.zeros(pos.shape[0], pos.shape[1], self.feats_dim, dtype=torch.float32, device=pos.device)
        coors = pos * mask.unsqueeze(2)

        for net in self.net:
            feats, coors = net(feats, coors, mask=mask)

            feats = feats * mask.unsqueeze(2)
            coors = coors * mask.unsqueeze(2)


        feats = torch.sum(feats * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)
        output = self.mlp(feats)

        return output
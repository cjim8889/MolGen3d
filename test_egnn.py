from re import L
import torch
from models.egnn import ModifiedPosEGNN
from models import CoorFlow
from models.classifier import PosClassifier

def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

# net = ModifiedPosEGNN(
#     in_dim = 3,
#     out_dim = 6,
#     m_dim=64,
#     fourier_features = 2,
#     num_nearest_neighbors = 6,
#     # dropout = 0.1,
#     norm_coors=True,
#     soft_edges=True
# )

net = CoorFlow(
    hidden_dim=128,
    gnn_size=3,
    block_size=8,
)

net.load_state_dict(
    torch.load(
        "model_checkpoint_qof5w8ec_90.pt",
        map_location="cpu"
    )['model_state_dict']
)

classifier = PosClassifier(
    feats_dim=64, hidden_dim=256, gnn_size=5
)

classifier.load_state_dict(torch.load("classifier.pt", map_location="cpu")['model_state_dict'])


with torch.no_grad():

    batch_size = 128
    base = torch.distributions.Normal(loc=0., scale=1.)

    z = base.sample(sample_shape=(batch_size, 29, 3))
    mask = torch.ones(batch_size, 29).to(torch.bool)
    mask_size = torch.randint(3, 29, (batch_size,))

    for idx in range(batch_size):
        mask[idx, mask_size[idx]:] = False

    z = z * mask.unsqueeze(2)
    z = remove_mean_with_mask(z, node_mask=mask)

    out, _ = net.inverse(z, mask=mask)

    pred = torch.sigmoid(classifier(out, mask=mask))


    print(torch.sum(pred > 0.5))
from models.pos.transformer import DenseTransformer
from models.pos.coupling import AffineCouplingFlow
import torch
n_dim = 3
hidden_dim = 16

num_layers_transformer = 4

ar_net = DenseTransformer(
                d_input=n_dim,
                d_output=n_dim * 2,
                d_model=hidden_dim,
                num_layers=num_layers_transformer,
                dim_feedforward=hidden_dim * 2,
                # dropout=0
            )

flow = AffineCouplingFlow(
    ar_net,
    scaling_func=torch.nn.Softplus(),
    split_dim=1,
    chunk_dim=2
)


x = torch.randn(1, 3, 18, dtype=torch.float32)

z, log_det = flow(x)
print(z)

x_re, _ = flow.inverse(z)
print(x)
print(x_re)
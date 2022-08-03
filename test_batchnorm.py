from models.pos.batchnorm import BatchNormFlow
import torch
# from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import Adam
from memory_profiler import profile

net = BatchNormFlow(
    num_features=3
)

optimiser = Adam(
    net.parameters(),
    lr=0.001
)


@profile
def func():
    x = torch.randn(128, 3, 29)

    net(x)

    for idx in range(2000):

        optimiser.zero_grad(set_to_none=True)
        z, log_det = net(x)

        loss = z.sum() + log_det.sum()
        loss.backward()

        optimiser.step()

if __name__ == "__main__":
    func()
    






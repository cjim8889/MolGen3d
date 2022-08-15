from sched import scheduler
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.optim import Adam


net = torch.nn.Linear(64, 64)

optimiser = Adam(
    net.parameters(),
    lr=0.001
)


print(optimiser.param_groups[0]['lr'])

scheduler = LinearLR(optimiser, start_factor=0.1, end_factor=1, total_iters=120)

for step in range(150):
    scheduler.step()
    print(optimiser.param_groups[0]['lr'])
 
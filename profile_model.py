import torch.autograd.profiler as profiler
import torch
from models.pos.flow import TransformerCoorFlow
from utils import get_datasets



model = TransformerCoorFlow(
    hidden_dim=32,
    block_size=6,
    max_nodes=18,
    num_layers_transformer=4,
    partition_size=(1, 9),
    conv1x1_node_wise=True,
    batch_norm=True
)

input = torch.randn(128, 3, 18)

model(input)

with torch.profiler.profile(with_stack=True, profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'), record_shapes=True) as prof:
    z, log_det = model(input)
    z.sum().backward()


# prof.export_chrome_trace("trace.json")
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_memory_usage', row_limit=25))



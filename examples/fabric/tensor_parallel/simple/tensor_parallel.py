import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

import lightning as L


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4096, 4096)
        self.w2 = nn.Linear(4096, 4096)
        self.w3 = nn.Linear(4096, 4096)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


fabric = L.Fabric(accelerator="cuda", devices=2)
fabric.launch()

model = FeedForward()

tp_mesh = init_device_mesh(fabric.device.type, (fabric.world_size,))
tp_plan = {
    "w1": ColwiseParallel(),
    "w2": ColwiseParallel(),
    "w3": RowwiseParallel(),
}
parallelize_module(model, tp_mesh, tp_plan)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Remove the setup call, otherwise DDP (default in Fabric) will interfere with TP
# model, optimizer = fabric.setup(model, optimizer)

# All inputs in a tensor-parallel group need to be the same
# Since we don't have a dataloader here, we can simulate this by
# setting the seed
L.seed_everything(0)


for i in range(10):
    input = torch.rand(8, 4096, device=fabric.device)
    output = model(input)
    loss = output.sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print("loss:", loss.item())

fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

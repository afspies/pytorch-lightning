from tkinter import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4096, 4096)
        self.w2 = nn.Linear(4096, 4096)
        self.w3 = nn.Linear(4096, 4096)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# Dummy launch to init distributed
fabric = L.Fabric(accelerator="cuda", devices=8)
fabric.launch()


mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]
dp_mesh = mesh["dp"]

fabric = L.Fabric(
    accelerator="cuda",
    devices=8,
    strategy=FSDPStrategy(device_mesh=mesh),
)
fabric._launched = True

model = FeedForward()

tp_plan = {
    "w1": ColwiseParallel(),
    "w2": ColwiseParallel(),
    "w3": RowwiseParallel(),
}
parallelize_module(model, tp_mesh, tp_plan)

model = fabric.setup(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
optimizer = fabric.setup_optimizers(optimizer)


for i in range(10):
    input = torch.rand(8, 4096, device=fabric.device)
    target = torch.rand(8, 4096, device=fabric.device)
    output = model(input)
    loss = F.l1_loss(output, target)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(loss.item())

fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
# fabric.print(torch.cuda.memory_summary())

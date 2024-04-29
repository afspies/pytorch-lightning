import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(1024, 1024)
        self.w1 = nn.Linear(1024, 1024)
        self.w1 = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


fabric = L.Fabric(
    accelerator="cuda",
    devices=2,
    # strategy=FSDPStrategy(),
)
fabric.launch()

fabric.seed_everything(42)

with fabric.rank_zero_first():
    dataset = WikiText2()

# 1B parameters
# model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)
model = FeedForward()


tp_mesh = init_device_mesh("cuda", (fabric.world_size,))
tp_plan = {
    "w1": ColwiseParallel(),
    "w2": RowwiseParallel(),
    "w3": ColwiseParallel(),
}
parallelize_module(model, tp_mesh, tp_plan)

model = fabric.setup(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
optimizer = fabric.setup_optimizers(optimizer)


for i in range(10):
    input, target = fabric.to_device(dataset[i])
    output = model(input.unsqueeze(0), target.unsqueeze(0))
    loss = F.nll_loss(output, target.view(-1))
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(loss.item())

fabric.print(torch.cuda.memory_summary())

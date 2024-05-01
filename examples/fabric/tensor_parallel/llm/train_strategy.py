
import lightning as L
import torch
import torch.nn.functional as F

from lightning.fabric.strategies import ModelParallelStrategy
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import Dataset, DataLoader

from model import ModelArgs, Transformer, parallelize


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_length: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __len__(self):
        return 64

    def __getitem__(self, item):
        return torch.randint(self.vocab_size, size=(self.seq_length, ))


fabric = L.Fabric(
    accelerator="cuda",
    devices="auto",
    strategy=ModelParallelStrategy(
        # Define the size of the 2D parallelism
        # Set to "auto" to apply TP intra-node and DP inter-node
        data_parallel_size=2,
        tensor_parallel_size=4,
    ),
)
fabric.launch()

device_mesh = fabric.strategy.device_mesh
# dp_mesh = device_mesh["data_parallel"]
# tp_mesh = device_mesh["tensor_parallel"]

model_args = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

with fabric.init_module():
    model = Transformer(model_args)

model = parallelize(model, fabric.strategy.device_mesh)
model.init_weights()

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=True)

# Set up model and optimizer
model, optimizer = fabric.setup(model, optimizer)

# All inputs in a tensor-parallel group need to be the same
# Since we don't have a dataloader here, we can simulate this by
# setting the seed
# L.seed_everything(dp_mesh.get_local_rank())

dataset = RandomTokenDataset(vocab_size=model_args.vocab_size, seq_length=129)
dataloader = DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

fabric.print("Starting training ...")

# Simplified training loop
for i, batch in enumerate(dataloader):
    # tokens = torch.randint(model_args.vocab_size, size=(batch_size, 129), device=fabric.device)
    inputs = batch[:, :-1]
    labels = batch[:, 1:]

    output = model(inputs)

    with loss_parallel():
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))

    fabric.backward(loss)
    optimizer.step()
    fabric.print(f"Iteration {i} complete")
    

# TODO
# Save a (distributed) checkpoint
# fabric.save("checkpoint.pt", {"model": model})

fabric.print("Training successfully completed!")
fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Avoid warning in PyTorch 2.4
torch.distributed.destroy_process_group()

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
        self.tokens = torch.randint(
            self.vocab_size, 
            size=(len(self), self.seq_length),
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self) -> int:
        return 128

    def __getitem__(self, item: int):
        return self.tokens[item]


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

# Access the device mesh if needed
device_mesh = fabric.strategy.device_mesh

# Initialize the model. TODO: Meta-device support
model_args = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)
with fabric.init_module():
    model = Transformer(model_args)

# Applies parallelization specific to the model (TP, FSDP2, activation checkpointing, ...)
model = parallelize(model, device_mesh)
model.init_weights()

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=True)

# Set up model and optimizer
model, optimizer = fabric.setup(model, optimizer)

dataset = RandomTokenDataset(vocab_size=model_args.vocab_size, seq_length=129)
dataloader = DataLoader(dataset, batch_size=8)

# Fabric configures the sampler automatically for you such that
# all batches in a tensor-parallel group are identical
dataloader = fabric.setup_dataloaders(dataloader)

# Simplified training loop
fabric.print("Starting training ...")

for i, batch in enumerate(dataloader):
    inputs = batch[:, :-1]
    labels = batch[:, 1:]

    output = model(inputs)

    with loss_parallel():
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))

    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(f"Iteration {i} complete")
    

# TODO
# Save a (distributed) checkpoint
# fabric.save("checkpoint.pt", {"model": model})

fabric.print("Training successfully completed!")
fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Avoid warning in PyTorch 2.4
torch.distributed.destroy_process_group()

import lightning as L
import torch
import torch.nn.functional as F

from lightning.fabric.strategies import ModelParallelStrategy
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader

from data import RandomTokenDataset
from model import ModelArgs, Transformer, parallelize


fabric = L.Fabric(
    accelerator="cuda",
    devices="auto",
    strategy=ModelParallelStrategy(
        # User-defined function that applies the desired parallelizations specific to the model
        # (TP, FSDP2, activation checkpointing, ...)
        parallelize_fn=parallelize,
        # Define the size of the 2D parallelism
        # Set to "auto" to apply TP intra-node and DP inter-node
        data_parallel_size=2,
        tensor_parallel_size=2,
    ),
)
fabric.launch()

# Initialize the model
model_args = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)
with fabric.init_module(empty_init=True):
    model = Transformer(model_args)
    
fabric.print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f} B")

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=True)

# Set up model and optimizer
model, optimizer = fabric.setup(model, optimizer)

# Materialize meta-device and init weights. TODO: Rewrite model with reset_parameters()
model.to_empty(device=fabric.device)
model.init_weights()

# Define dataset/dataloader
dataset = RandomTokenDataset(vocab_size=model_args.vocab_size, seq_length=2048)
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


# Save a (distributed) checkpoint
# See `fabric consolidate --help` if you need to convert the checkpoint to a single file
state = {"model": model, "optimizer": optimizer, "iteration": i}
fabric.save("checkpoint.pt", state)


fabric.print("Training successfully completed!")
fabric.print(f"Peak memory usage: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

# Avoid warning in PyTorch 2.4
torch.distributed.destroy_process_group()

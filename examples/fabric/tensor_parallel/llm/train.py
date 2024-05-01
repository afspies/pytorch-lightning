# Code dapted from official PyTorch example at
# https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)

from model import ModelArgs, Transformer


# Dummy launch to init distributed
fabric = L.Fabric(accelerator="cuda", devices="auto")
fabric.launch()

# Define the tensor parallel size
tensor_parallel_size = 4
fsdp_size = fabric.world_size // tensor_parallel_size

# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
device_mesh = init_device_mesh("cuda", (fsdp_size, tensor_parallel_size), mesh_dim_names=("dp", "tp"))

fabric.print(f"Device Mesh created: {device_mesh=}")
tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

fabric = L.Fabric(
    accelerator="cuda",
    devices="auto",
    strategy=FSDPStrategy(device_mesh=dp_mesh),
)
fabric._launched = True


model_args = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

with fabric.init_module():
    model = Transformer(model_args)

# Parallelize the first embedding and the last linear out projection
plan = {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(),
    ),
    "output": ColwiseParallel(
        input_layouts=Shard(1),
        output_layouts=Replicate()
    ),
    "norm": SequenceParallel(),
    "layers.0": PrepareModuleInput(
        input_layouts=(Replicate(), None),
        desired_input_layouts=(Shard(1), None),
        use_local_output=True,
    ),
}
model = parallelize_module(model, tp_mesh, plan)

# Parallelize each transformer block
for transformer_block in model.layers:
    plan = {
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "attention_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
        "ffn_norm": SequenceParallel(),
    }

    # Adjust attention module to use the local number of heads
    attn_layer = transformer_block.attention
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

    # Apply the plan for the current transformer block
    parallelize_module(transformer_block, tp_mesh, plan)


# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=True)

# Set up model and optimizer
model, optimizer = fabric.setup(model, optimizer)

# All inputs in a tensor-parallel group need to be the same
# Since we don't have a dataloader here, we can simulate this by
# setting the seed
L.seed_everything(dp_mesh.get_local_rank())

fabric.print("Starting training ...")
num_iterations = 10
batch_size = 8

# Simplified training loop
for i in range(num_iterations):
    tokens = torch.randint(model_args.vocab_size, size=(batch_size, model_args.dim), device=fabric.device)
    output = model(tokens)
    loss = output.sum()
    fabric.backward(loss)
    optimizer.step()
    fabric.print(f"Iteration {i} complete")

fabric.print("Training successfully completed!")
fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

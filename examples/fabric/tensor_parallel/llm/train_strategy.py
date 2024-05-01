# Code dapted from official PyTorch example at
# https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism

import lightning as L
import torch
import torch.nn.functional as F

from lightning.fabric.strategies import ModelParallelStrategy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)

from model import ModelArgs, Transformer, TransformerBlock

# Define the tensor parallel size
# tensor_parallel_size = 4
# fsdp_size = fabric.world_size // tensor_parallel_size


fabric = L.Fabric(
    accelerator="cuda",
    devices="auto",
    strategy=ModelParallelStrategy(
        data_parallel_size=2,
        tensor_parallel_size=4,
    ),
)
fabric.launch()

device_mesh = fabric.strategy.device_mesh
dp_mesh = device_mesh["data_parallel"]
tp_mesh = device_mesh["tensor_parallel"]

model_args = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

with fabric.init_module():
    model = Transformer(model_args)
model.init_weights()

# Parallelize the first embedding and the last linear out projection
plan = {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(),
    ),
    "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
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
    tokens = torch.randint(model_args.vocab_size, size=(batch_size, 129), device=fabric.device)
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]

    output = model(inputs)

    with loss_parallel():
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))

    fabric.backward(loss)
    optimizer.step()
    fabric.print(f"Iteration {i} complete")
    

# Save a (distributed) checkpoint
# fabric.save("checkpoint.pt", {"model": model})

fabric.print("Training successfully completed!")
fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

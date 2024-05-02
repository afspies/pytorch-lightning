# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override

from fabric.strategies.fsdp import _distributed_checkpoint_save, _distributed_checkpoint_load
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import (
    TBroadcast,
    _apply_filter,
    _BackwardSyncControl,
    _Sharded,
    _validate_keys_for_strict_loading,
)
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_2_1,
    _TORCH_GREATER_EQUAL_2_2,
    _TORCH_GREATER_EQUAL_2_3,
)
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, _Stateful

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    _POLICY = Union[Set[Type[Module]], Callable[[Module, bool, int], bool], ModuleWrapPolicy]
    _SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]

_FSDP_ALIASES = ("fsdp", "fsdp_cpu_offload")


class ModelParallelStrategy(ParallelStrategy):
    """Enables user-defined parallelism applied to a model.

    Currently supports up to 2D parallelism, for example Fully Sharded Data-Parallel combined with
    Tensor Parallelism.

    Arguments:
        parallelize_fn: A function that applies parallelisms to a module. The strategy will provide the
            model and device mesh as input.
        data_parallel_size: The number of devices within a data-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of nodes in the cluster.
        tensor_parallel_size: The number of devices within a tensor-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of GPUs in a single node.
    """

    def __init__(
        self,
        parallelize_fn: Callable[[Module, DeviceMesh], Module],
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size:  Union[Literal["auto"], int] = "auto",
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        super().__init__()
        self._parallelize_fn = parallelize_fn
        self._data_parallel_size = data_parallel_size
        self._tensor_parallel_size = tensor_parallel_size
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        # self._backward_sync_control = _FSDPBackwardSyncControl()

        self._device_mesh: Optional[DeviceMesh] = None

    @property
    def device_mesh(self) -> DeviceMesh:
        if self._device_mesh is None:
            raise RuntimeError()  # TODO
        return self._device_mesh

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(f"The `{type(self).__name__}` does not use the `CheckpointIO` plugin interface.")

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: CheckpointIO) -> None:
        raise NotImplementedError(f"The `{type(self).__name__}` does not support setting a `CheckpointIO` plugin.")

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        assert self.device_mesh is not None
        data_parallel_mesh = self.device_mesh["data_parallel"]
        return {"num_replicas": data_parallel_mesh.size(), "rank": data_parallel_mesh.get_local_rank()}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()
        self._setup_device_mesh()

    @override
    def setup_module(self, module: Module) -> Module:
        module = self._parallelize_fn(module, self.device_mesh)
        # TODO
        # _move_torchmetrics_to_device(module, self.root_device)
        return module

    @override
    def module_to_device(self, module: Module) -> None:
        # TODO
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> ContextManager:
        precision_init_ctx = self.precision.module_init_context()
        # module_sharded_ctx = self.module_sharded_context()
        empty_ctx = _EmptyInit(enabled=bool(empty_init))
        stack = ExitStack()
        if _TORCH_GREATER_EQUAL_2_1 and empty_init:
            # TODO
            # Materialization happens in `setup`. When modules get wrapped by FSDP, the sequence of operations is:
            # 1) materialize module 2) call `reset_parameters()` 3) shard the module.
            # These operations are applied to each submodule 'bottom up' in the module hierarchy.
            stack.enter_context(torch.device("meta"))
        else:
            stack.enter_context(empty_ctx)
        stack.enter_context(precision_init_ctx)
        # stack.enter_context(module_sharded_ctx)
        return stack

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.root_device.index])
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        raise NotImplementedError()

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:

        """Save model, optimizer, and other state to a checkpoint on disk.

        If the state-dict-type is ``'full'``, the checkpoint will be written to a single file containing the weights,
        optimizer state and other metadata. If the state-dict-type is ``'sharded'``, the checkpoint gets saved as a
        directory containing one file per process, with model- and optimizer shards stored per file. Additionally, it
        creates a metadata file `meta.pt` with the rest of the user's state (only saved from rank 0).

        """
        if storage_options is not None:
            raise TypeError(
                "`ModelParallel.save_checkpoint(..., storage_options=...)` is not supported because"
                " `ModelParallel` does not use the `CheckpointIO`."
            )
        # TODO:
        # if filter is not None and self._state_dict_type == "sharded":
        #     # https://github.com/pytorch/pytorch/issues/105379
        #     raise NotImplementedError(
        #         "ModelParallel doesn't support loading sharded filtered checkpoints, so saving them is disabled."
        #     )
        _distributed_checkpoint_save(state, path)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        # TODO:
        _distributed_checkpoint_load(state, path)

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _setup_device_mesh(self):
        self._data_parallel_size = self.num_nodes if self._data_parallel_size == "auto" else self._data_parallel_size
        self._tensor_parallel_size = self.num_processes if self._tensor_parallel_size == "auto" else self._tensor_parallel_size
        # TODO: Error message
        assert self._data_parallel_size * self._tensor_parallel_size == self.world_size
        self._device_mesh = init_device_mesh(
            device_type=self.root_device.type,
            mesh_shape=(self._data_parallel_size, self._tensor_parallel_size),
            mesh_dim_names=("data_parallel", "tensor_parallel")
        )

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

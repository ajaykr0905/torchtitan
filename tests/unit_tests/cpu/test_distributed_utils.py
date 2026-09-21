# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.checkpoint import checkpoint

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import set_spmd_meshes, spmd_dense_sp_enabled
from torchtitan.distributed.utils import init_distributed


def test_bf16x9_is_enabled_on_future_nvidia_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matmul = SimpleNamespace(fp32_precision="ieee")
    monkeypatch.setattr(dist_utils, "device_type", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)

    dist_utils.enable_fp32_matmul_emulation_with_bf16x9()

    assert matmul.fp32_precision == "bfx9"


def test_fake_pg_uses_requested_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "6")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torchtitan.distributed.utils.init_fake_mode") as init_fake_mode,
    ):
        assert init_distributed(CommConfig(mode="fake_backend")) == 8
    init_fake_mode.assert_called_once_with(8, rank=6)


def test_fake_pg_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "8")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(ValueError, match=r"RANK must be in \[0, 8\)"),
    ):
        init_distributed(CommConfig(mode="fake_backend"))


def test_dist_sum_tensor_keeps_local_result_as_tensor():
    value = torch.tensor(3, dtype=torch.int64)

    result = dist_utils.dist_sum_tensor(value)

    assert result is value


def test_dist_sum_tensor_waits_for_distributed_result():
    value = torch.tensor(3, dtype=torch.int64)
    reduced = torch.tensor(8, dtype=torch.int64)
    mesh = cast(DeviceMesh, object())

    with (
        patch.object(dist_utils.funcol, "all_reduce", return_value=reduced) as reduce,
        patch.object(dist_utils.funcol, "wait_tensor", return_value=reduced) as wait,
    ):
        result = dist_utils.dist_sum_tensor(value, mesh)

    assert result is reduced
    reduce.assert_called_once_with(value, reduceOp="SUM", group=mesh)
    wait.assert_called_once_with(reduced)


@pytest.mark.parametrize("enable_sequence_parallel", [False, True])
def test_spmd_context_exposes_dense_sp_state(
    enable_sequence_parallel: bool,
) -> None:
    dense_mesh = cast(DeviceMesh, object())
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=2,
        pp=1,
        ep=1,
        world_size=2,
        enable_sequence_parallel=enable_sequence_parallel,
    )
    parallel_dims._single_axis_meshes["tp"] = dense_mesh

    with (
        patch.object(parallel_dims, "spmd_dense_mesh", return_value=dense_mesh),
        patch.object(parallel_dims, "spmd_sparse_mesh", return_value=None),
        patch(
            "torchtitan.distributed.spmd_types.set_current_spmd_mesh",
            return_value=contextlib.nullcontext(),
        ),
        patch(
            "torchtitan.distributed.spmd_types.spmd_dense_mesh",
            return_value=dense_mesh,
        ),
        dist_utils.get_spmd_context(parallel_dims=parallel_dims),
    ):
        assert spmd_dense_sp_enabled() is enable_sequence_parallel


def test_dense_sp_state_compiles_with_checkpoint() -> None:
    dense_mesh = cast(DeviceMesh, object())
    set_spmd_meshes(
        dense_mesh=dense_mesh,
        sparse_mesh=None,
        dense_sp_enabled=True,
    )

    def checkpointed_forward(input):
        def forward(value):
            assert spmd_dense_sp_enabled()
            return value + 1

        return checkpoint(forward, input, use_reentrant=False)

    compiled_forward = torch.compile(
        checkpointed_forward,
        backend="eager",
        fullgraph=True,
    )
    input = torch.randn(2, 3, requires_grad=True)

    output = compiled_forward(input)
    output.sum().backward()

    torch.testing.assert_close(output, input + 1)
    set_spmd_meshes(
        dense_mesh=dense_mesh,
        sparse_mesh=None,
        dense_sp_enabled=False,
    )


def test_mean_flops_tensor_uses_float64_sum_and_device_division():
    device = torch.device("cpu")
    mesh = cast(DeviceMesh, object())
    extra_pg = cast(torch.distributed.ProcessGroup, object())
    reduced = torch.tensor(84.0, dtype=torch.float64, device=device)
    tensor_constructor = torch.tensor

    with (
        patch.object(
            dist_utils.torch,
            "tensor",
            wraps=tensor_constructor,
        ) as make_tensor,
        patch.object(
            dist_utils,
            "dist_sum_tensor",
            return_value=reduced,
        ) as reduce,
        patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("unexpected host materialization"),
        ),
        patch.object(
            torch.Tensor,
            "tolist",
            side_effect=AssertionError("unexpected host materialization"),
        ),
    ):
        result = dist_utils.mean_flops_tensor(
            21,
            device=device,
            mesh=mesh,
            divisor=4,
            extra_pg=extra_pg,
        )

    make_tensor.assert_called_once_with(21, dtype=torch.float64, device=device)
    reduce.assert_called_once()
    assert reduce.call_args.args[0].dtype is torch.float64
    assert reduce.call_args.args[0].device == device
    assert reduce.call_args.kwargs == {"mesh": mesh, "extra_pg": extra_pg}
    torch.testing.assert_close(result, torch.tensor(21.0, dtype=torch.float64))


def test_materialize_scalar_tensors_packs_one_host_read():
    values = [torch.tensor(2), torch.tensor([3.5])]
    original_tolist = torch.Tensor.tolist
    num_tolist_calls = 0

    def record_tolist(value):
        nonlocal num_tolist_calls
        num_tolist_calls += 1
        return original_tolist(value)

    with patch.object(torch.Tensor, "tolist", record_tolist):
        result = dist_utils.materialize_scalar_tensors(values)

    assert result == (2.0, 3.5)
    assert num_tolist_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_materialize_scalar_tensors_supports_mixed_devices():
    values = [
        torch.tensor(2.0, device="cuda"),
        torch.tensor(3),
        torch.tensor([4.5], device="cuda"),
    ]
    original_tolist = torch.Tensor.tolist
    num_cuda_tolist_calls = 0

    def record_tolist(value):
        nonlocal num_cuda_tolist_calls
        if value.device.type == "cuda":
            num_cuda_tolist_calls += 1
        return original_tolist(value)

    with patch.object(torch.Tensor, "tolist", record_tolist):
        result = dist_utils.materialize_scalar_tensors(values)

    assert result == (2.0, 3.0, 4.5)
    assert num_cuda_tolist_calls == 1

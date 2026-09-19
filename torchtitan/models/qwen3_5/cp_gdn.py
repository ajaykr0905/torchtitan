# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Gated DeltaNet stages backed by Attention Gym."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from attn_gym.linear import l2norm
from attn_gym.linear.context_parallel import (
    context_parallel_conv_history,
    ContextParallelRouting,
)
from attn_gym.linear.gdn.context_parallel import context_parallel_gdn

from torchtitan.distributed.context_parallel import ContextParallelPartitioner
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.cp_attention import LoadBalancedCPInnerAttention

from .gdn import GatedDeltaNetMetadata, InnerGatedDeltaNet


class ContextParallelInnerGatedDeltaNet(
    LoadBalancedCPInnerAttention, InnerGatedDeltaNet
):
    """GatedDeltaNet with distributed convolution and recurrent-state routing."""

    @dataclass(kw_only=True, slots=True)
    class Config(LoadBalancedCPInnerAttention.Config, InnerGatedDeltaNet.Config):
        pass

    @classmethod
    def cp_shard_metadata(
        cls,
        input_dict: dict[str, Any],
        partitioner: ContextParallelPartitioner,
        config: LoadBalancedCPInnerAttention.Config,
    ) -> dict[str, Any]:
        """Attach rank-local GDN routing for the current CP partition."""
        assert isinstance(config, cls.Config)
        tokens_T = input_dict.get("input")
        if not isinstance(tokens_T, torch.Tensor):
            raise ValueError("GatedDeltaNet CP requires tensor input.")

        context_metadata = input_dict.get("attention_masks")
        if context_metadata is None:
            num_tokens = tokens_T.shape[0] * partitioner.cp_mesh.size(0)
            cu_seqlens_global = [0, num_tokens]
            gdn_metadata = GatedDeltaNetMetadata(varlen=None)
            context_metadata = {
                "quadratic_attention": None,
                "deltanet": gdn_metadata,
            }
        else:
            if not isinstance(context_metadata, Mapping):
                raise ValueError(
                    "GatedDeltaNet CP requires attention_masks to contain "
                    "GatedDeltaNetMetadata."
                )
            gdn_metadata = context_metadata.get("deltanet")
            if not isinstance(gdn_metadata, GatedDeltaNetMetadata):
                raise ValueError(
                    "GatedDeltaNet CP requires GatedDeltaNetMetadata in "
                    "attention_masks['deltanet']."
                )
            if gdn_metadata.varlen is None:
                num_tokens = tokens_T.shape[0] * partitioner.cp_mesh.size(0)
                cu_seqlens_global = [0, num_tokens]
            else:
                cu_seqlens_global = gdn_metadata.varlen.cu_seq_q.tolist()

        group = partitioner.cp_mesh.get_group()
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=cu_seqlens_global,
            fragments=partitioner.token_partition(cu_seqlens_global[-1]),
            cp_rank=dist.get_rank(group),
            device=tokens_T.device,
            conv_history=config.conv_kernel_size - 1,
        )
        input_dict["attention_masks"] = {
            **context_metadata,
            "deltanet": replace(gdn_metadata, cp_routing=routing),
        }
        return input_dict

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        routing: ContextParallelRouting | None = None,
    ) -> torch.Tensor:
        if routing is None:
            raise ValueError("GatedDeltaNet context parallelism requires routing.")
        if routing.tail_sources.shape[1] != conv_q_weight_C1W.shape[-1] - 1:
            raise ValueError(
                "GatedDeltaNet CP routing convolution history must match the "
                "model's convolution width minus one."
            )
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            a_TH,
            b_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_H,
            routing.cu_seqlens,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            routing=routing,
        )

    def short_convolution(
        self,
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        assert routing is not None, "CP forward must validate routing."
        assert (
            initial_state is None
        ), "GatedDeltaNet context parallelism constructs convolution history."
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "GatedDeltaNet context parallelism requires an active "
                "multi-rank CP mesh axis."
            )
        initial_state = context_parallel_conv_history(
            x_TC.unsqueeze(0), routing, cp_group
        )
        return super().short_convolution(
            x_TC,
            weight_C1W,
            initial_state,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )

    def gdn_core(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        # The CP kernel reads rank-local sequence boundaries from routing.
        del cu_seqlens
        assert routing is not None, "CP forward must validate routing."
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "GatedDeltaNet context parallelism requires an active "
                "multi-rank CP mesh axis."
            )
        normalized_q_1THK = l2norm(  # pyrefly: ignore [not-callable]
            xq_THK.unsqueeze(0), cu_seqlens=routing.cu_seqlens
        )
        normalized_k_1THK = l2norm(  # pyrefly: ignore [not-callable]
            xk_THK.unsqueeze(0), cu_seqlens=routing.cu_seqlens
        )
        output_1THV, _ = context_parallel_gdn(
            normalized_q_1THK,
            normalized_k_1THK,
            xv_THV.unsqueeze(0),
            g_TH.unsqueeze(0),
            beta_TH.unsqueeze(0),
            routing=routing,
            group=cp_group,
            scale=xq_THK.shape[-1] ** -0.5,
        )
        return output_1THV.squeeze(0)

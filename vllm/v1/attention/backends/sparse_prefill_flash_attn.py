# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend with a PyTorch sparse-prefill override."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.sparse_prefill_utils import (
    SparsePrefillTopKConfig,
    get_sparse_prefill_topk_config,
    is_full_prefill_request,
    run_sparse_prefill_attention,
)

logger = init_logger(__name__)


class SparsePrefillFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "SPARSE_PREFILL_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparsePrefillFlashAttentionImpl"]:
        return SparsePrefillFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder


class SparsePrefillFlashAttentionImpl(FlashAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        cfg = get_sparse_prefill_topk_config()
        if cfg is None:
            raise ValueError(
                "Sparse prefill FlashAttention backend was selected without a "
                "valid sparse-prefill configuration."
            )
        self.sparse_cfg: SparsePrefillTopKConfig = cfg
        logger.info_once(
            "Sparse prefill FlashAttention backend enabled with scheme %s "
            "(q_block=%d, k_block=%d, topk=%d). Only uncached full-prefill "
            "requests use the PyTorch sparse path; decode remains dense "
            "FlashAttention.",
            self.sparse_cfg.key,
            self.sparse_cfg.q_block,
            self.sparse_cfg.k_block,
            self.sparse_cfg.topk,
            scope="local",
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dense_output = super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        if attn_metadata is None or self.attn_type != AttentionType.DECODER:
            return dense_output
        if key is None or value is None:
            return dense_output
        if attn_metadata.max_query_len <= 1:
            return dense_output
        if not attn_metadata.causal or attn_metadata.use_cascade:
            return dense_output
        if self.dcp_world_size != 1:
            return dense_output
        if self.alibi_slopes is not None or self.sinks is not None:
            return dense_output
        if self.sliding_window not in (None, (-1, -1)):
            return dense_output

        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        if num_actual_tokens <= 0:
            return dense_output

        query_start_loc = [
            int(x)
            for x in attn_metadata.query_start_loc.detach().to(device="cpu").tolist()
        ]
        seq_lens = [
            int(x) for x in attn_metadata.seq_lens.detach().to(device="cpu").tolist()
        ]

        query_actual = query[:num_actual_tokens]
        key_actual = key[:num_actual_tokens]
        value_actual = value[:num_actual_tokens]

        overwritten_reqs = 0
        for req_idx, seq_len in enumerate(seq_lens):
            q_start = query_start_loc[req_idx]
            q_end = query_start_loc[req_idx + 1]
            query_len = q_end - q_start
            if not is_full_prefill_request(query_len, seq_len):
                continue

            sparse_output = run_sparse_prefill_attention(
                query=query_actual[q_start:q_end],
                key=key_actual[q_start:q_end],
                value=value_actual[q_start:q_end],
                scaling=self.scale,
                cfg=self.sparse_cfg,
                output_dtype=dense_output.dtype,
            )
            dense_output[q_start:q_end].copy_(sparse_output)
            overwritten_reqs += 1

        if overwritten_reqs > 0:
            logger.info_once(
                "Sparse prefill FlashAttention override is active. Full-prefill "
                "requests are recomputed with the PyTorch sparse path; cached-prefix "
                "prefill and decode requests remain dense.",
                scope="local",
            )

        return dense_output

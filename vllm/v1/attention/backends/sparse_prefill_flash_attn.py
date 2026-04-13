# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend with a Triton sparse-prefill override."""

from __future__ import annotations

import os

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
    AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV,
    AUTOPTQ_VLLM_SPARSE_IMPL_ENV,
    SparsePrefillTopKConfig,
    gather_full_sequence_kv_from_paged_cache,
    get_sparse_prefill_topk_config,
    get_sparse_prefill_impl_mode,
    get_sparse_prefill_retain_score_log_mode,
    is_cached_prefix_prefill_request,
    is_full_prefill_request,
    is_sparse_prefill_request,
    run_sparse_prefill_attention,
)
from vllm.v1.attention.ops.triton_sparse_prefill import (
    run_triton_sparse_prefill_attention,
)

logger = init_logger(__name__)

AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV = (
    "AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL"
)


def _parse_optional_env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    logger.warning_once(
        "Sparse prefill FlashAttention received unrecognized %s=%r; using "
        "default value %s.",
        name,
        value,
        default,
        scope="local",
    )
    return default


def _ensure_sparse_output_has_no_nan(
    sparse_output: torch.Tensor,
    *,
    req_idx: int,
    query_len: int,
    seq_len: int,
    request_kind: str,
    impl_mode: str,
) -> None:
    nan_mask = torch.isnan(sparse_output)
    if not bool(nan_mask.any().item()):
        return

    nan_count = int(nan_mask.sum().item())
    raise RuntimeError(
        "Sparse prefill FlashAttention produced NaN values in output "
        f"(request={req_idx}, query_len={query_len}, seq_len={seq_len}, "
        f"request_kind={request_kind}, impl={impl_mode}, nan_count={nan_count})."
    )


def _build_request_token_indices(
    requests: list[dict[str, int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    spans = [
        torch.arange(
            request["q_start"],
            request["q_end"],
            device=device,
            dtype=torch.long,
        )
        for request in requests
    ]
    if not spans:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(spans, dim=0)


def _build_subset_metadata(
    attn_metadata: FlashAttentionMetadata,
    *,
    req_indices: torch.Tensor,
    token_indices: torch.Tensor,
) -> FlashAttentionMetadata:
    seq_lens = attn_metadata.seq_lens.index_select(0, req_indices)
    query_lens = torch.empty_like(seq_lens)
    query_starts = attn_metadata.query_start_loc.index_select(0, req_indices)
    query_ends = attn_metadata.query_start_loc.index_select(0, req_indices + 1)
    query_lens.copy_(query_ends - query_starts)

    new_query_start_loc = torch.zeros(
        req_indices.numel() + 1,
        dtype=attn_metadata.query_start_loc.dtype,
        device=attn_metadata.query_start_loc.device,
    )
    if query_lens.numel() > 0:
        new_query_start_loc[1:] = torch.cumsum(query_lens, dim=0)

    if attn_metadata.block_table is not None and attn_metadata.block_table.numel() > 0:
        block_table = attn_metadata.block_table.index_select(0, req_indices)
    else:
        block_table = torch.empty(
            (req_indices.numel(), 0),
            dtype=torch.int32,
            device=attn_metadata.query_start_loc.device,
        )

    if attn_metadata.slot_mapping is not None and attn_metadata.slot_mapping.numel() > 0:
        slot_mapping = attn_metadata.slot_mapping.index_select(0, token_indices)
    else:
        slot_mapping = torch.empty(
            (token_indices.numel(),),
            dtype=torch.int64,
            device=attn_metadata.query_start_loc.device,
        )

    dcp_context_kv_lens = None
    max_dcp_context_kv_len = None
    if attn_metadata.dcp_context_kv_lens is not None:
        dcp_context_kv_lens = attn_metadata.dcp_context_kv_lens.index_select(
            0, req_indices
        )
        if dcp_context_kv_lens.numel() > 0:
            max_dcp_context_kv_len = int(dcp_context_kv_lens.max().item())

    return FlashAttentionMetadata(
        num_actual_tokens=int(token_indices.numel()),
        max_query_len=int(query_lens.max().item()) if query_lens.numel() > 0 else 0,
        query_start_loc=new_query_start_loc,
        max_seq_len=int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=max_dcp_context_kv_len,
        dcp_context_kv_lens=dcp_context_kv_lens,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=attn_metadata.max_num_splits,
        causal=attn_metadata.causal,
    )


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
        self.sparse_impl_mode = get_sparse_prefill_impl_mode()
        self.retain_score_log_mode = get_sparse_prefill_retain_score_log_mode()
        self.record_retain_score = self.retain_score_log_mode != "off"
        self.force_paged_full_prefill = _parse_optional_env_flag(
            AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV,
            default=True,
        )
        logger.info_once(
            "Sparse prefill FlashAttention backend enabled with scheme %s "
            "(q_block=%d, k_block=%d, topk=%d, impl=%s). Eligible prefill "
            "requests use the sparse override path while decode remains dense "
            "FlashAttention.",
            self.sparse_cfg.key,
            self.sparse_cfg.q_block,
            self.sparse_cfg.k_block,
            self.sparse_cfg.topk,
            self.sparse_impl_mode,
            scope="local",
        )
        if self.record_retain_score:
            logger.info_once(
                "Sparse prefill FlashAttention will log retain score stats "
                "with mode=%s for sparse requests because %s is enabled.",
                self.retain_score_log_mode,
                AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV,
                scope="local",
            )
        if self.force_paged_full_prefill:
            logger.info_once(
                "Sparse prefill FlashAttention will route full-prefill sparse "
                "requests through paged KV metadata by default. Set %s=0 to "
                "keep the contiguous full-prefill path.",
                AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV,
                scope="local",
            )
        else:
            logger.info_once(
                "Sparse prefill FlashAttention will keep the contiguous "
                "full-prefill path because %s disables paged full-prefill "
                "routing.",
                AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV,
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
        if attn_metadata is None or self.attn_type != AttentionType.DECODER:
            return super().forward(
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
        if key is None or value is None:
            return super().forward(
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
        if attn_metadata.max_query_len <= 1:
            return super().forward(
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
        if not attn_metadata.causal or attn_metadata.use_cascade:
            return super().forward(
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
        if self.dcp_world_size != 1:
            return super().forward(
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
        if self.alibi_slopes is not None or self.sinks is not None:
            return super().forward(
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
        if self.sliding_window not in (None, (-1, -1)):
            return super().forward(
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

        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        if num_actual_tokens <= 0:
            return output

        query_start_loc = [
            int(x)
            for x in attn_metadata.query_start_loc.detach().to(device="cpu").tolist()
        ]
        seq_lens = [
            int(x) for x in attn_metadata.seq_lens.detach().to(device="cpu").tolist()
        ]
        block_table = attn_metadata.block_table

        query_actual = query[:num_actual_tokens]
        key_actual = key[:num_actual_tokens]
        value_actual = value[:num_actual_tokens]

        sparse_requests: list[dict[str, int]] = []
        dense_requests: list[dict[str, int]] = []
        for req_idx, seq_len in enumerate(seq_lens):
            q_start = query_start_loc[req_idx]
            q_end = query_start_loc[req_idx + 1]
            query_len = q_end - q_start
            request = {
                "req_idx": req_idx,
                "seq_len": seq_len,
                "q_start": q_start,
                "q_end": q_end,
                "query_len": query_len,
            }
            if is_sparse_prefill_request(query_len, seq_len):
                sparse_requests.append(request)
            else:
                dense_requests.append(request)

        if not sparse_requests:
            return super().forward(
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

        if dense_requests:
            dense_req_indices = torch.tensor(
                [request["req_idx"] for request in dense_requests],
                device=attn_metadata.seq_lens.device,
                dtype=torch.long,
            )
            dense_token_indices = _build_request_token_indices(
                dense_requests,
                device=query.device,
            )
            dense_metadata = _build_subset_metadata(
                attn_metadata,
                req_indices=dense_req_indices,
                token_indices=dense_token_indices,
            )
            dense_query = query_actual.index_select(0, dense_token_indices)
            dense_key = key_actual.index_select(0, dense_token_indices)
            dense_value = value_actual.index_select(0, dense_token_indices)
            dense_subset_output = torch.empty(
                dense_query.shape,
                dtype=output.dtype,
                device=output.device,
            )
            super().forward(
                layer=layer,
                query=dense_query,
                key=dense_key,
                value=dense_value,
                kv_cache=kv_cache,
                attn_metadata=dense_metadata,
                output=dense_subset_output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
            output.index_copy_(0, dense_token_indices, dense_subset_output)

        used_triton = 0
        used_pytorch_fallback = 0
        used_pytorch_direct = 0
        cached_prefix_reqs = 0
        for request in sparse_requests:
            req_idx = request["req_idx"]
            q_start = request["q_start"]
            q_end = request["q_end"]
            query_len = request["query_len"]
            seq_len = request["seq_len"]
            request_kind = (
                "full_prefill"
                if is_full_prefill_request(query_len, seq_len)
                else "cached_prefix"
            )
            sparse_query = query_actual[q_start:q_end]
            sparse_output = None

            if self.sparse_impl_mode != "torch":
                try:
                    use_paged_triton_path = (
                        block_table is not None
                        and block_table.numel() > 0
                        and (
                            is_cached_prefix_prefill_request(query_len, seq_len)
                            or (
                                self.force_paged_full_prefill
                                and is_full_prefill_request(query_len, seq_len)
                            )
                        )
                    )
                    if use_paged_triton_path:
                        sparse_output = run_triton_sparse_prefill_attention(
                            query=sparse_query,
                            scaling=self.scale,
                            cfg=self.sparse_cfg,
                            output_dtype=output.dtype,
                            kv_cache=kv_cache,
                            block_table_row=block_table[req_idx],
                            seq_len=seq_len,
                            kv_cache_dtype=self.kv_cache_dtype,
                            record_retain_score=self.record_retain_score,
                            retain_score_log_mode=self.retain_score_log_mode,
                            layer=layer,
                        )
                    elif is_full_prefill_request(query_len, seq_len):
                        sparse_output = run_triton_sparse_prefill_attention(
                            query=sparse_query,
                            key=key_actual[q_start:q_end],
                            value=value_actual[q_start:q_end],
                            scaling=self.scale,
                            cfg=self.sparse_cfg,
                            output_dtype=output.dtype,
                            record_retain_score=self.record_retain_score,
                            retain_score_log_mode=self.retain_score_log_mode,
                            layer=layer,
                        )
                    if sparse_output is not None:
                        _ensure_sparse_output_has_no_nan(
                            sparse_output,
                            req_idx=req_idx,
                            query_len=query_len,
                            seq_len=seq_len,
                            request_kind=request_kind,
                            impl_mode="triton",
                        )
                except (RuntimeError, NotImplementedError, ValueError) as exc:
                    if self.sparse_impl_mode == "triton":
                        raise RuntimeError(
                            "Sparse prefill FlashAttention was configured with "
                            f"{AUTOPTQ_VLLM_SPARSE_IMPL_ENV}=triton, but the "
                            f"Triton sparse path failed for request {req_idx} "
                            f"(query_len={query_len}, seq_len={seq_len}): {exc}"
                        ) from exc
                    logger.warning_once(
                        "Sparse prefill FlashAttention Triton path fell back to the "
                        "PyTorch sparse reference path (%s).",
                        exc,
                        scope="local",
                    )

            if sparse_output is None:
                if self.sparse_impl_mode == "triton":
                    raise RuntimeError(
                        "Sparse prefill FlashAttention was configured with "
                        f"{AUTOPTQ_VLLM_SPARSE_IMPL_ENV}=triton, but this sparse "
                        f"request could not be served by the Triton path "
                        f"(request={req_idx}, query_len={query_len}, seq_len={seq_len})."
                    )
                if is_full_prefill_request(query_len, seq_len):
                    sparse_key = key_actual[q_start:q_end]
                    sparse_value = value_actual[q_start:q_end]
                elif is_cached_prefix_prefill_request(query_len, seq_len):
                    if block_table is None or block_table.numel() == 0:
                        logger.warning_once(
                            "Sparse prefill FlashAttention skipped cached-prefix "
                            "recompute because block tables were unavailable. "
                            "Falling back to dense output for that request.",
                            scope="local",
                        )
                        dense_single_output = torch.empty(
                            (query_len, self.num_heads, self.head_size),
                            dtype=output.dtype,
                            device=output.device,
                        )
                        single_req_indices = torch.tensor(
                            [req_idx],
                            device=attn_metadata.seq_lens.device,
                            dtype=torch.long,
                        )
                        single_token_indices = torch.arange(
                            q_start,
                            q_end,
                            device=query.device,
                            dtype=torch.long,
                        )
                        single_metadata = _build_subset_metadata(
                            attn_metadata,
                            req_indices=single_req_indices,
                            token_indices=single_token_indices,
                        )
                        super().forward(
                            layer=layer,
                            query=query_actual[q_start:q_end],
                            key=key_actual[q_start:q_end],
                            value=value_actual[q_start:q_end],
                            kv_cache=kv_cache,
                            attn_metadata=single_metadata,
                            output=dense_single_output,
                            output_scale=output_scale,
                            output_block_scale=output_block_scale,
                        )
                        output[q_start:q_end].copy_(dense_single_output)
                        continue
                    try:
                        sparse_key, sparse_value = gather_full_sequence_kv_from_paged_cache(
                            kv_cache=kv_cache,
                            block_table_row=block_table[req_idx],
                            seq_len=seq_len,
                            num_kv_heads=self.num_kv_heads,
                            head_dim=self.head_size,
                            kv_cache_dtype=self.kv_cache_dtype,
                        )
                    except (NotImplementedError, ValueError) as exc:
                        logger.warning_once(
                            "Sparse prefill FlashAttention skipped cached-prefix "
                            "PyTorch fallback because paged-KV reconstruction was "
                            "unavailable (%s). Falling back to dense output for "
                            "that request.",
                            exc,
                            scope="local",
                        )
                        dense_single_output = torch.empty(
                            (query_len, self.num_heads, self.head_size),
                            dtype=output.dtype,
                            device=output.device,
                        )
                        single_req_indices = torch.tensor(
                            [req_idx],
                            device=attn_metadata.seq_lens.device,
                            dtype=torch.long,
                        )
                        single_token_indices = torch.arange(
                            q_start,
                            q_end,
                            device=query.device,
                            dtype=torch.long,
                        )
                        single_metadata = _build_subset_metadata(
                            attn_metadata,
                            req_indices=single_req_indices,
                            token_indices=single_token_indices,
                        )
                        super().forward(
                            layer=layer,
                            query=query_actual[q_start:q_end],
                            key=key_actual[q_start:q_end],
                            value=value_actual[q_start:q_end],
                            kv_cache=kv_cache,
                            attn_metadata=single_metadata,
                            output=dense_single_output,
                            output_scale=output_scale,
                            output_block_scale=output_block_scale,
                        )
                        output[q_start:q_end].copy_(dense_single_output)
                        continue
                    cached_prefix_reqs += 1
                else:
                    continue

                sparse_output = run_sparse_prefill_attention(
                    query=sparse_query,
                    key=sparse_key,
                    value=sparse_value,
                    scaling=self.scale,
                    cfg=self.sparse_cfg,
                    output_dtype=output.dtype,
                    record_retain_score=self.record_retain_score,
                    retain_score_log_mode=self.retain_score_log_mode,
                    layer=layer,
                )
                _ensure_sparse_output_has_no_nan(
                    sparse_output,
                    req_idx=req_idx,
                    query_len=query_len,
                    seq_len=seq_len,
                    request_kind=request_kind,
                    impl_mode="torch",
                )
                if self.sparse_impl_mode == "torch":
                    used_pytorch_direct += 1
                else:
                    used_pytorch_fallback += 1
            else:
                used_triton += 1
                if is_cached_prefix_prefill_request(query_len, seq_len):
                    cached_prefix_reqs += 1

            output[q_start:q_end].copy_(sparse_output)

        if used_triton > 0:
            logger.info_once(
                "Sparse prefill FlashAttention override is active. Eligible "
                "prefill requests, including cached-prefix prefill, are "
                "recomputed with the Triton sparse path; decode remains dense.",
                scope="local",
            )
        if used_pytorch_fallback > 0:
            logger.info_once(
                "Sparse prefill FlashAttention used the PyTorch sparse reference "
                "fallback for at least one eligible request.",
                scope="local",
            )
        if used_pytorch_direct > 0:
            logger.info_once(
                "Sparse prefill FlashAttention is using the PyTorch sparse "
                f"reference path because {AUTOPTQ_VLLM_SPARSE_IMPL_ENV}=torch.",
                scope="local",
            )
        if cached_prefix_reqs > 0:
            logger.info_once(
                "Sparse prefill FlashAttention is using paged KV metadata for "
                "cached-prefix prefill requests.",
                scope="local",
            )

        return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import RCP_LN2
from vllm.v1.attention.backends.sparse_prefill_utils import (
    SparsePrefillRetainScoreLogMode,
    SparsePrefillSelectionStats,
    SparsePrefillTopKConfig,
    _keep_block_mask_to_block_indices,
    _merge_mandatory_and_sparse_blocks,
    _mean_pool_attention_blocks,
    _pool_query_scores_to_blocks,
    build_sparse_prefill_selection_stats,
    format_sparse_prefill_selection_policy,
    format_sparse_prefill_per_head_payload,
    resolve_sparse_prefill_layer_info,
    resolve_sparse_prefill_retain_score_log_mode,
    should_log_sparse_prefill_layer_info,
    should_log_sparse_prefill_per_head_stats,
)

logger = init_logger(__name__)
AUTOPTQ_VLLM_SPARSE_TIMING_BREAKDOWN_ENV = (
    "AUTOPTQ_VLLM_SPARSE_TIMING_BREAKDOWN"
)
AUTOPTQ_VLLM_SPARSE_DEBUG_FULLY_MASKED_ROWS_ENV = (
    "AUTOPTQ_VLLM_SPARSE_DEBUG_FULLY_MASKED_ROWS"
)
RETAIN_TILE_SIZE = 128
RETAIN_MASK_WORD_BITS = 32
RETAIN_MASK_WORDS = RETAIN_TILE_SIZE // RETAIN_MASK_WORD_BITS


def _parse_env_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_sparse_prefill_timing_breakdown_enabled() -> bool:
    return _parse_env_flag(os.getenv(AUTOPTQ_VLLM_SPARSE_TIMING_BREAKDOWN_ENV))


def _is_sparse_prefill_fully_masked_row_debug_enabled() -> bool:
    return _parse_env_flag(os.getenv(AUTOPTQ_VLLM_SPARSE_DEBUG_FULLY_MASKED_ROWS_ENV))


@dataclass
class SparsePrefillTimingBreakdown:
    enabled: bool
    device: torch.device
    timings_ms: dict[str, float]

    def add(self, name: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        self.timings_ms[name] = self.timings_ms.get(name, 0.0) + float(elapsed_ms)

    def sorted_items(self) -> list[tuple[str, float]]:
        return sorted(
            self.timings_ms.items(),
            key=lambda item: item[1],
            reverse=True,
        )


def _timed_call(
    breakdown: SparsePrefillTimingBreakdown | None,
    name: str,
    fn,
    /,
    *args,
    **kwargs,
):
    if breakdown is None or not breakdown.enabled:
        return fn(*args, **kwargs)

    if breakdown.device.type == "cuda":
        torch.cuda.synchronize(device=breakdown.device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if breakdown.device.type == "cuda":
        torch.cuda.synchronize(device=breakdown.device)
    breakdown.add(name, (time.perf_counter() - start) * 1000.0)
    return result


def _log_sparse_prefill_timing_breakdown(
    *,
    layer: object | None,
    query_len: int,
    seq_len: int,
    paged_kv: bool,
    total_ms: float,
    breakdown: SparsePrefillTimingBreakdown | None,
) -> None:
    if breakdown is None or not breakdown.enabled or not breakdown.timings_ms:
        return

    layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)
    details = ", ".join(
        (
            f"{name}={elapsed_ms:.3f}ms "
            f"({(elapsed_ms / total_ms * 100.0) if total_ms > 0 else 0.0:.1f}%)"
        )
        for name, elapsed_ms in breakdown.sorted_items()
    )
    logger.info(
        "Sparse prefill Triton timing breakdown: layer_idx=%s layer_name=%s "
        "q_len=%d seq_len=%d path=%s total=%.3fms breakdown=[%s].",
        layer_idx if layer_idx is not None else "NA",
        layer_name or "unknown",
        int(query_len),
        int(seq_len),
        "paged" if paged_kv else "contiguous",
        float(total_ms),
        details,
    )


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _repeat_kv_heads(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    )


@dataclass(frozen=True)
class SparseTopKBlockMetadata:
    topk_block_indices: torch.Tensor
    topk_block_counts: torch.Tensor
    q_blocks: int
    k_blocks: int
    q_abs_offset: int
    retain_tile_block_mask: torch.Tensor | None = None
    selection_stats: SparsePrefillSelectionStats | None = None


def _extract_unquantized_paged_kv(
    kv_cache: torch.Tensor,
    *,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.dim() != 5 or kv_cache.shape[0] != 2:
        raise NotImplementedError(
            "Triton sparse prefill currently expects flash-style paged KV cache "
            f"with shape [2, num_blocks, block_size, num_kv_heads, head_dim], got {tuple(kv_cache.shape)}."
        )
    if str(kv_cache_dtype).startswith("fp8"):
        raise NotImplementedError(
            "Triton sparse prefill does not yet support fp8 KV cache."
        )
    key_cache, value_cache = kv_cache.unbind(0)
    return key_cache.contiguous(), value_cache.contiguous()


def _pool_paged_key_blocks(
    *,
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    cfg: SparsePrefillTopKConfig,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")
    cache_block_size = int(key_cache.shape[1])
    num_required_blocks = _ceil_div(seq_len, cache_block_size)
    if block_table_row.numel() < num_required_blocks:
        raise ValueError(
            "Sparse Triton prefill requires enough block-table entries to read "
            f"seq_len={seq_len}, cache_block_size={cache_block_size}, "
            f"got {block_table_row.numel()} entries."
        )

    device = key_cache.device
    block_table_row = block_table_row.to(device=device, dtype=torch.long).reshape(-1)
    token_positions = torch.arange(seq_len, device=device, dtype=torch.long)
    block_ids = block_table_row.index_select(
        0,
        torch.div(token_positions, cache_block_size, rounding_mode="floor"),
    )
    slots = block_ids * cache_block_size + token_positions.remainder(cache_block_size)
    flat_key = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])
    dense_key = flat_key.index_select(0, slots)
    dense_key_states = dense_key.transpose(0, 1).unsqueeze(0)
    return _mean_pool_attention_blocks(dense_key_states, cfg.k_block)


def _build_causal_valid_block_mask(
    *,
    q_len: int,
    k_len: int,
    q_block: int,
    k_block: int,
    device: torch.device,
) -> torch.Tensor:
    if q_len <= 0 or k_len <= 0:
        raise ValueError(f"Expected positive q_len/k_len, got {q_len}/{k_len}.")
    q_blocks = _ceil_div(q_len, q_block)
    k_blocks = _ceil_div(k_len, k_block)
    q_last_token = (
        torch.arange(q_blocks, device=device, dtype=torch.long) + 1
    ) * q_block - 1
    q_last_token = q_last_token.clamp_max(q_len - 1)
    q_abs_max = q_last_token + (k_len - q_len)
    k_block_start = torch.arange(k_blocks, device=device, dtype=torch.long) * k_block
    return k_block_start.view(1, 1, 1, k_blocks) <= q_abs_max.view(1, 1, q_blocks, 1)


def _build_causal_valid_token_block_mask(
    *,
    q_abs_start: int,
    q_count: int,
    k_len: int,
    k_block: int,
    device: torch.device,
) -> torch.Tensor:
    if q_count <= 0 or k_len <= 0:
        raise ValueError(f"Expected positive q_count/k_len, got {q_count}/{k_len}.")
    k_blocks = _ceil_div(k_len, k_block)
    q_abs = torch.arange(q_count, device=device, dtype=torch.long) + int(q_abs_start)
    k_block_start = torch.arange(k_blocks, device=device, dtype=torch.long) * k_block
    return k_block_start.view(1, 1, 1, k_blocks) <= q_abs.view(1, 1, q_count, 1)


def _pad_topk_indices(
    topk_idx: torch.Tensor,
    counts: torch.Tensor,
    *,
    full_topk: int,
) -> torch.Tensor:
    padded = topk_idx.to(torch.int32)
    if padded.shape[-1] < full_topk:
        pad_shape = list(padded.shape)
        pad_shape[-1] = full_topk - padded.shape[-1]
        pad = torch.full(
            pad_shape,
            -1,
            dtype=padded.dtype,
            device=padded.device,
        )
        padded = torch.cat([padded, pad], dim=-1)
    rank = torch.arange(full_topk, device=padded.device, dtype=torch.int32).view(
        *([1] * (padded.dim() - 1)),
        full_topk,
    )
    invalid = rank >= counts.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(padded, -1), padded)


def _build_retain_tile_block_mask(
    topk_block_indices: torch.Tensor,
    *,
    seq_len: int,
    k_block: int,
) -> torch.Tensor:
    if topk_block_indices.dim() != 3:
        raise ValueError(
            "Expected topk_block_indices to have shape [heads, q_blocks, topk], "
            f"got {tuple(topk_block_indices.shape)}."
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")
    if k_block <= 0:
        raise ValueError(f"k_block must be > 0, got {k_block}.")

    num_heads, q_blocks, _ = topk_block_indices.shape
    num_k_iters = _ceil_div(seq_len, RETAIN_TILE_SIZE)
    block_mask = torch.zeros(
        num_heads,
        q_blocks,
        num_k_iters,
        RETAIN_MASK_WORDS,
        device=topk_block_indices.device,
        dtype=torch.int64,
    )

    valid = topk_block_indices >= 0
    if not bool(valid.any().item()):
        return block_mask.to(torch.int32)

    head_idx, q_block_idx, topk_rank = valid.nonzero(as_tuple=True)
    del topk_rank
    block_idx = topk_block_indices[valid].to(torch.int64)
    block_start = block_idx * int(k_block)
    block_end = (block_start + int(k_block) - 1).clamp_max(seq_len - 1)
    first_iter = torch.div(block_start, RETAIN_TILE_SIZE, rounding_mode="floor")
    last_iter = torch.div(block_end, RETAIN_TILE_SIZE, rounding_mode="floor")
    bit_values = (1 << torch.arange(
        RETAIN_MASK_WORD_BITS,
        device=topk_block_indices.device,
        dtype=torch.int64,
    ))
    flat_mask = block_mask.view(-1, RETAIN_MASK_WORDS)
    max_tile_span = _ceil_div(int(k_block) + RETAIN_TILE_SIZE - 1, RETAIN_TILE_SIZE)

    for span in range(max_tile_span):
        iter_idx = first_iter + span
        active = iter_idx <= last_iter
        if not bool(active.any().item()):
            continue
        active_head = head_idx[active]
        active_q_block = q_block_idx[active]
        active_iter = iter_idx[active]
        active_block = block_idx[active]
        tile_sparse_base = torch.div(
            active_iter * RETAIN_TILE_SIZE,
            int(k_block),
            rounding_mode="floor",
        )
        local_block_idx = active_block - tile_sparse_base
        word_idx = torch.div(
            local_block_idx,
            RETAIN_MASK_WORD_BITS,
            rounding_mode="floor",
        )
        bit_idx = torch.remainder(local_block_idx, RETAIN_MASK_WORD_BITS)
        flat_row_idx = (
            (active_head * q_blocks + active_q_block) * num_k_iters + active_iter
        )
        flat_mask.index_put_(
            (flat_row_idx, word_idx),
            bit_values.index_select(0, bit_idx),
            accumulate=True,
        )

    return block_mask.to(torch.int32)


def _summarize_topk_block_selection(
    *,
    masked_scores: torch.Tensor,
    valid_block_mask: torch.Tensor,
    keep_block_mask: torch.Tensor,
) -> SparsePrefillSelectionStats:
    valid_block_counts = valid_block_mask.sum(dim=-1).to(torch.int32)
    kept_block_counts = keep_block_mask.sum(dim=-1).to(torch.int32)
    valid_row_mask = valid_block_counts > 0
    safe_logsumexp = torch.logsumexp(masked_scores, dim=-1)
    safe_logsumexp = torch.where(
        valid_row_mask,
        safe_logsumexp,
        torch.zeros_like(safe_logsumexp),
    )

    retained_attention_score_mass = torch.where(
        keep_block_mask & valid_row_mask.unsqueeze(-1),
        torch.exp(masked_scores - safe_logsumexp.unsqueeze(-1)),
        torch.zeros_like(masked_scores),
    ).sum(dim=-1)

    return build_sparse_prefill_selection_stats(
        valid_block_counts=valid_block_counts,
        kept_block_counts=kept_block_counts,
        valid_row_mask=valid_row_mask,
        retained_attention_score_mass=retained_attention_score_mass,
    )


def build_sparse_topk_block_metadata(
    *,
    query: torch.Tensor,
    pooled_key: torch.Tensor,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    k_len: int,
    build_retain_tile_block_mask: bool = False,
    record_selection_stats: bool = False,
    breakdown: SparsePrefillTimingBreakdown | None = None,
) -> SparseTopKBlockMetadata:
    query_states = query.transpose(0, 1).unsqueeze(0)
    if pooled_key.shape[1] != query_states.shape[1]:
        if query_states.shape[1] % pooled_key.shape[1] != 0:
            raise ValueError(
                "Sparse Triton prefill requires query heads to be divisible "
                f"by KV heads, got num_heads={query_states.shape[1]} and "
                f"num_kv_heads={pooled_key.shape[1]}."
            )
        pooled_key = _timed_call(
            breakdown,
            "topk_repeat_kv_heads",
            _repeat_kv_heads,
            pooled_key,
            query_states.shape[1] // pooled_key.shape[1],
        )
    if cfg.q_pooling == "mean_after":

        def _build_mean_after_block_scores() -> tuple[torch.Tensor, torch.Tensor]:
            pooled_key_t = pooled_key.to(torch.float32).transpose(-1, -2)
            q_abs_offset = k_len - query.shape[0]
            block_score_rows: list[torch.Tensor] = []
            valid_block_rows: list[torch.Tensor] = []

            # Stream one q-block at a time so sparse_test_28 does not
            # materialize the full token-score matrix.
            for q_start in range(0, query.shape[0], cfg.q_block):
                q_end = min(q_start + int(cfg.q_block), query.shape[0])
                query_block = query_states[:, :, q_start:q_end, :].to(torch.float32)
                token_scores = torch.matmul(query_block, pooled_key_t) * float(scaling)
                token_valid_mask = _build_causal_valid_token_block_mask(
                    q_abs_start=q_start + q_abs_offset,
                    q_count=q_end - q_start,
                    k_len=k_len,
                    k_block=cfg.k_block,
                    device=query.device,
                )
                if token_valid_mask.shape[1] != token_scores.shape[1]:
                    token_valid_mask = token_valid_mask.expand(
                        token_valid_mask.shape[0],
                        token_scores.shape[1],
                        token_valid_mask.shape[2],
                        token_valid_mask.shape[3],
                    )
                block_scores_row, valid_block_row = _pool_query_scores_to_blocks(
                    token_scores,
                    token_valid_mask,
                    q_block=cfg.q_block,
                    q_pooling=cfg.q_pooling,
                )
                block_score_rows.append(block_scores_row)
                valid_block_rows.append(valid_block_row)

            return torch.cat(block_score_rows, dim=2), torch.cat(valid_block_rows, dim=2)

        block_scores, valid_block_mask = _timed_call(
            breakdown,
            "topk_mean_after",
            _build_mean_after_block_scores,
        )
    elif cfg.q_pooling == "mean_before":
        pooled_query = _timed_call(
            breakdown,
            "topk_query_pool",
            _mean_pool_attention_blocks,
            query_states,
            cfg.q_block,
        )
        block_scores = _timed_call(
            breakdown,
            "topk_score_matmul",
            lambda: torch.matmul(
                pooled_query.to(torch.float32),
                pooled_key.to(torch.float32).transpose(-1, -2),
            ) * float(scaling),
        )
        valid_block_mask = _timed_call(
            breakdown,
            "topk_build_mask",
            _build_causal_valid_block_mask,
            q_len=query.shape[0],
            k_len=k_len,
            q_block=cfg.q_block,
            k_block=cfg.k_block,
            device=query.device,
        )
        if valid_block_mask.shape[1] != block_scores.shape[1]:
            valid_block_mask = valid_block_mask.expand(
                valid_block_mask.shape[0],
                block_scores.shape[1],
                valid_block_mask.shape[2],
                valid_block_mask.shape[3],
            )
    else:
        raise ValueError(f"Unsupported sparse q_pooling mode: {cfg.q_pooling!r}.")

    masked_scores = _timed_call(
        breakdown,
        "topk_apply_mask",
        lambda: torch.where(
            valid_block_mask,
            block_scores,
            torch.full_like(block_scores, float("-inf")),
        ),
    )
    block_probs = None
    if cfg.threshold is not None:
        block_probs = _timed_call(
            breakdown,
            "topk_softmax",
            lambda: torch.softmax(masked_scores, dim=-1),
        )
        row_has_valid = valid_block_mask.any(dim=-1, keepdim=True)
        block_probs = torch.where(
            row_has_valid,
            block_probs,
            torch.zeros_like(block_probs),
        )
    keep_block_mask, counts = _timed_call(
        breakdown,
        "topk_select",
        _merge_mandatory_and_sparse_blocks,
        masked_scores,
        valid_block_mask,
        cfg=cfg,
        block_probs=block_probs,
    )
    selection_width = int(counts.max().item()) if counts.numel() > 0 else 0
    padded_topk_idx = _timed_call(
        breakdown,
        "topk_pad_indices",
        _keep_block_mask_to_block_indices,
        keep_block_mask,
        counts=counts,
        full_topk=selection_width,
    )
    topk_block_indices = padded_topk_idx.squeeze(0).contiguous()
    topk_block_counts = counts.squeeze(0).contiguous()
    retain_tile_block_mask = (
        _timed_call(
            breakdown,
            "retain_build_tile_mask",
            _build_retain_tile_block_mask,
            topk_block_indices,
            seq_len=k_len,
            k_block=int(cfg.k_block),
        )
        if build_retain_tile_block_mask
        else None
    )
    selection_stats = (
        _timed_call(
            breakdown,
            "topk_selection_stats",
            _summarize_topk_block_selection,
            masked_scores=masked_scores,
            valid_block_mask=valid_block_mask,
            keep_block_mask=keep_block_mask,
        )
        if record_selection_stats
        else None
    )
    return SparseTopKBlockMetadata(
        topk_block_indices=topk_block_indices,
        topk_block_counts=topk_block_counts,
        q_blocks=_ceil_div(query.shape[0], cfg.q_block),
        k_blocks=_ceil_div(k_len, cfg.k_block),
        q_abs_offset=k_len - query.shape[0],
        retain_tile_block_mask=retain_tile_block_mask,
        selection_stats=selection_stats,
    )


def build_sparse_topk_block_metadata_from_paged_cache(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    scaling: float,
    kv_cache_dtype: str,
    cfg: SparsePrefillTopKConfig,
    build_retain_tile_block_mask: bool = False,
    record_selection_stats: bool = False,
    breakdown: SparsePrefillTimingBreakdown | None = None,
) -> SparseTopKBlockMetadata:
    key_cache, _ = _timed_call(
        breakdown,
        "paged_extract_kv_for_topk",
        _extract_unquantized_paged_kv,
        kv_cache,
        kv_cache_dtype=kv_cache_dtype,
    )
    pooled_key = _timed_call(
        breakdown,
        "paged_pool_key",
        _pool_paged_key_blocks,
        key_cache=key_cache,
        block_table_row=block_table_row,
        seq_len=seq_len,
        cfg=cfg,
    )
    return build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=scaling,
        cfg=cfg,
        k_len=seq_len,
        build_retain_tile_block_mask=build_retain_tile_block_mask,
        record_selection_stats=record_selection_stats,
        breakdown=breakdown,
    )


@triton.jit
def _smallk_weighted_value_sum(
    probs,
    value,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    if BLOCK_N == 1:
        return probs.to(tl.float32) * value.to(tl.float32)

    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    lane_ids = tl.arange(0, BLOCK_N)
    probs_f32 = probs.to(tl.float32)
    value_f32 = value.to(tl.float32)
    for lane_idx in tl.static_range(0, BLOCK_N):
        lane_mask = lane_ids == lane_idx
        prob_lane = tl.sum(tl.where(lane_mask[None, :], probs_f32, 0.0), axis=1)
        value_lane = tl.sum(tl.where(lane_mask[:, None], value_f32, 0.0), axis=0)
        acc += prob_lane[:, None] * value_lane[None, :]
    return acc


def _fully_masked_row_mask(
    *,
    topk_metadata: SparseTopKBlockMetadata,
    q_len: int,
    q_block: int,
    k_block: int,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(q_len, device=device, dtype=torch.int64)
    q_block_idx = torch.div(q_positions, q_block, rounding_mode="floor")
    selected = topk_metadata.topk_block_indices.to(device=device, dtype=torch.int64).index_select(
        1,
        q_block_idx,
    )
    q_abs = q_positions + int(topk_metadata.q_abs_offset)
    selected_start = torch.where(selected >= 0, selected * int(k_block), -1)
    has_valid = (
        (selected >= 0)
        & (selected_start <= q_abs.view(1, q_len, 1))
    ).any(dim=-1)
    return (~has_valid).transpose(0, 1).contiguous()


def _fully_masked_row_mask_for_k1(
    *,
    topk_metadata: SparseTopKBlockMetadata,
    q_len: int,
    q_block: int,
    device: torch.device,
) -> torch.Tensor:
    return _fully_masked_row_mask(
        topk_metadata=topk_metadata,
        q_len=q_len,
        q_block=q_block,
        k_block=1,
        device=device,
    )


def _log_fully_masked_row_debug(
    *,
    topk_metadata: SparseTopKBlockMetadata,
    q_len: int,
    seq_len: int,
    cfg: SparsePrefillTopKConfig,
    paged_kv: bool,
    layer: object | None,
) -> None:
    if not _is_sparse_prefill_fully_masked_row_debug_enabled():
        return

    fully_masked = _fully_masked_row_mask(
        topk_metadata=topk_metadata,
        q_len=q_len,
        q_block=int(cfg.q_block),
        k_block=int(cfg.k_block),
        device=topk_metadata.topk_block_indices.device,
    )
    masked_rows = int(fully_masked.sum().item())
    total_rows = int(fully_masked.numel())
    masked_token_rows = fully_masked.any(dim=1)
    masked_tokens = int(masked_token_rows.sum().item())
    sample_payload = "[]"

    if masked_rows > 0:
        q_positions = torch.arange(q_len, device=fully_masked.device, dtype=torch.int64)
        q_block_idx = torch.div(q_positions, int(cfg.q_block), rounding_mode="floor")
        q_abs = q_positions + int(topk_metadata.q_abs_offset)
        sample_parts: list[str] = []
        masked_pos, masked_heads = fully_masked.nonzero(as_tuple=True)
        sample_count = min(8, int(masked_pos.numel()))
        for sample_idx in range(sample_count):
            q_pos = int(masked_pos[sample_idx].item())
            head_idx = int(masked_heads[sample_idx].item())
            q_blk = int(q_block_idx[q_pos].item())
            keep_count = int(topk_metadata.topk_block_counts[head_idx, q_blk].item())
            selected_blocks = [
                int(v)
                for v in topk_metadata.topk_block_indices[
                    head_idx, q_blk, :keep_count
                ].tolist()
            ]
            sample_parts.append(
                "q=%d abs=%d h=%d qb=%d sel=%s"
                % (
                    q_pos,
                    int(q_abs[q_pos].item()),
                    head_idx,
                    q_blk,
                    selected_blocks,
                )
            )
        sample_payload = "[" + "; ".join(sample_parts) + "]"

    layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)
    selection_policy = format_sparse_prefill_selection_policy(cfg)
    logger.info(
        "Sparse prefill fully-masked-row debug: layer_idx=%s layer_name=%s "
        "q_len=%d seq_len=%d path=%s q_block=%d k_block=%d %s "
        "masked_rows=%d/%d masked_tokens=%d/%d sample=%s.",
        layer_idx if layer_idx is not None else "NA",
        layer_name or "unknown",
        int(q_len),
        int(seq_len),
        "paged" if paged_kv else "contiguous",
        int(cfg.q_block),
        int(cfg.k_block),
        selection_policy,
        masked_rows,
        total_rows,
        masked_tokens,
        int(q_len),
        sample_payload,
    )


def _gather_paged_value_mean(
    *,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    cache_block_size = int(value_cache.shape[1])
    num_required_blocks = _ceil_div(seq_len, cache_block_size)
    if block_table_row.numel() < num_required_blocks:
        raise ValueError(
            "Sparse Triton prefill requires enough block-table entries to read "
            f"seq_len={seq_len}, cache_block_size={cache_block_size}, "
            f"got {block_table_row.numel()} entries."
        )

    device = value_cache.device
    block_table_row = block_table_row.to(device=device, dtype=torch.long).reshape(-1)
    token_positions = torch.arange(seq_len, device=device, dtype=torch.long)
    block_ids = block_table_row.index_select(
        0,
        torch.div(token_positions, cache_block_size, rounding_mode="floor"),
    )
    slots = block_ids * cache_block_size + token_positions.remainder(cache_block_size)
    flat_value = value_cache.reshape(-1, value_cache.shape[2], value_cache.shape[3])
    return flat_value.index_select(0, slots).to(torch.float32).mean(dim=0)


def _apply_k1_reference_fallback(
    *,
    output: torch.Tensor,
    value_mean: torch.Tensor,
    topk_metadata: SparseTopKBlockMetadata,
    q_block: int,
) -> torch.Tensor:
    fully_masked = _fully_masked_row_mask_for_k1(
        topk_metadata=topk_metadata,
        q_len=int(output.shape[0]),
        q_block=q_block,
        device=output.device,
    )
    if not bool(fully_masked.any().item()):
        return output

    num_heads = int(output.shape[1])
    num_kv_heads = int(value_mean.shape[0])
    num_queries_per_kv = num_heads // num_kv_heads
    kv_head_idx = torch.div(
        torch.arange(num_heads, device=output.device, dtype=torch.long),
        num_queries_per_kv,
        rounding_mode="floor",
    )
    mean_per_head = value_mean.to(device=output.device, dtype=output.dtype).index_select(
        0,
        kv_head_idx,
    )
    return torch.where(
        fully_masked.unsqueeze(-1),
        mean_per_head.unsqueeze(0),
        output,
    )


@triton.jit
def _sparse_prefill_attention_contiguous_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_indices_ptr,
    topk_counts_ptr,
    output_ptr,
    sm_scale,
    q_stride_tok,
    q_stride_head,
    q_stride_dim,
    k_stride_tok,
    k_stride_head,
    k_stride_dim,
    v_stride_tok,
    v_stride_head,
    v_stride_dim,
    topk_head_stride,
    topk_qblock_stride,
    topk_topk_stride,
    count_head_stride,
    count_qblock_stride,
    out_stride_tok,
    out_stride_head,
    out_stride_dim,
    q_len,
    seq_len,
    q_abs_offset,
    num_queries_per_kv,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_SMALL_K: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_queries_per_kv

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    q_pos = q_block_idx * Q_BLOCK + offs_m
    q_mask = (offs_m < Q_BLOCK) & (q_pos < q_len)
    safe_q_pos = tl.where(q_mask, q_pos, 0)
    q_abs = safe_q_pos + q_abs_offset
    dim_mask = offs_d < HEAD_SIZE

    q_offsets = (
        safe_q_pos[:, None] * q_stride_tok
        + head_idx * q_stride_head
        + offs_d[None, :] * q_stride_dim
    )
    query = tl.load(
        query_ptr + q_offsets,
        mask=q_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    running_denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    running_out = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    selected_count = tl.load(
        topk_counts_ptr + head_idx * count_head_stride + q_block_idx * count_qblock_stride
    )

    for topk_rank in range(TOPK):
        topk_active = topk_rank < selected_count
        block_idx = tl.load(
            topk_indices_ptr
            + head_idx * topk_head_stride
            + q_block_idx * topk_qblock_stride
            + topk_rank * topk_topk_stride
        )
        block_active = topk_active & (block_idx >= 0)
        raw_k_abs = tl.where(block_active, block_idx, 0) * K_BLOCK + offs_n
        token_valid = block_active & (offs_n < K_BLOCK) & (raw_k_abs < seq_len)
        k_abs = tl.where(token_valid, raw_k_abs, 0)

        k_offsets = (
            k_abs[None, :] * k_stride_tok
            + kv_head_idx * k_stride_head
            + offs_d[:, None] * k_stride_dim
        )
        key = tl.load(
            key_ptr + k_offsets,
            mask=dim_mask[:, None] & token_valid[None, :],
            other=0.0,
        )

        if USE_SMALL_K and BLOCK_N == 1:
            key_vec = tl.sum(key.to(tl.float32), axis=1)
            score_vec = tl.sum(
                query.to(tl.float32) * tl.expand_dims(key_vec, 0),
                axis=1,
            ) * sm_scale
            scores = tl.expand_dims(score_vec, 1)
        else:
            scores = tl.dot(query, key) * sm_scale
        attn_mask = (
            q_mask[:, None]
            & token_valid[None, :]
            & (q_abs[:, None] >= k_abs[None, :])
        )
        masked_scores = tl.where(attn_mask, scores, float("-inf"))
        row_has_valid = tl.max(attn_mask.to(tl.int32), axis=1) > 0
        block_max = tl.max(masked_scores, axis=1)
        safe_block_max = tl.where(row_has_valid, block_max, 0.0)
        new_max = tl.where(
            row_has_valid,
            tl.maximum(running_max, block_max),
            running_max,
        )
        # Avoid exp2(-inf - -inf) before a row has seen any valid block.
        prev_scale = tl.where(
            running_max == float("-inf"),
            0.0,
            tl.math.exp2(running_max - new_max),
        )
        block_scale = tl.where(
            row_has_valid,
            tl.math.exp2(safe_block_max - new_max),
            0.0,
        )
        probs = tl.where(
            attn_mask,
            tl.math.exp2(masked_scores - safe_block_max[:, None]),
            0.0,
        )

        v_offsets = (
            k_abs[:, None] * v_stride_tok
            + kv_head_idx * v_stride_head
            + offs_d[None, :] * v_stride_dim
        )
        value = tl.load(
            value_ptr + v_offsets,
            mask=token_valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        if USE_SMALL_K:
            value_acc = _smallk_weighted_value_sum(
                probs,
                value,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
            )
        else:
            value_acc = tl.dot(probs.to(value.dtype), value)
        running_out = (
            running_out * prev_scale[:, None]
            + value_acc * block_scale[:, None]
        )
        running_denom = (
            running_denom * prev_scale
            + tl.sum(probs, axis=1) * block_scale
        )
        running_max = new_max

    output = running_out / tl.maximum(running_denom[:, None], 1e-20)
    out_offsets = (
        safe_q_pos[:, None] * out_stride_tok
        + head_idx * out_stride_head
        + offs_d[None, :] * out_stride_dim
    )
    tl.store(
        output_ptr + out_offsets,
        output,
        mask=q_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _sparse_prefill_attention_paged_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    topk_indices_ptr,
    topk_counts_ptr,
    output_ptr,
    sm_scale,
    q_stride_tok,
    q_stride_head,
    q_stride_dim,
    kc_stride_blk,
    kc_stride_slot,
    kc_stride_head,
    kc_stride_dim,
    vc_stride_blk,
    vc_stride_slot,
    vc_stride_head,
    vc_stride_dim,
    topk_head_stride,
    topk_qblock_stride,
    topk_topk_stride,
    count_head_stride,
    count_qblock_stride,
    out_stride_tok,
    out_stride_head,
    out_stride_dim,
    q_len,
    seq_len,
    q_abs_offset,
    num_queries_per_kv,
    CACHE_BLOCK_SIZE: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_SMALL_K: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_queries_per_kv

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    q_pos = q_block_idx * Q_BLOCK + offs_m
    q_mask = (offs_m < Q_BLOCK) & (q_pos < q_len)
    safe_q_pos = tl.where(q_mask, q_pos, 0)
    q_abs = safe_q_pos + q_abs_offset
    dim_mask = offs_d < HEAD_SIZE

    q_offsets = (
        safe_q_pos[:, None] * q_stride_tok
        + head_idx * q_stride_head
        + offs_d[None, :] * q_stride_dim
    )
    query = tl.load(
        query_ptr + q_offsets,
        mask=q_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    running_denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    running_out = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    selected_count = tl.load(
        topk_counts_ptr + head_idx * count_head_stride + q_block_idx * count_qblock_stride
    )

    for topk_rank in range(TOPK):
        topk_active = topk_rank < selected_count
        block_idx = tl.load(
            topk_indices_ptr
            + head_idx * topk_head_stride
            + q_block_idx * topk_qblock_stride
            + topk_rank * topk_topk_stride
        )
        block_active = topk_active & (block_idx >= 0)
        raw_k_abs = tl.where(block_active, block_idx, 0) * K_BLOCK + offs_n
        token_valid = block_active & (offs_n < K_BLOCK) & (raw_k_abs < seq_len)
        k_abs = tl.where(token_valid, raw_k_abs, 0)
        physical_block_idx = tl.load(
            block_table_ptr + (k_abs // CACHE_BLOCK_SIZE),
            mask=token_valid,
            other=0,
        )
        slot = k_abs % CACHE_BLOCK_SIZE

        k_offsets = (
            physical_block_idx[None, :] * kc_stride_blk
            + slot[None, :] * kc_stride_slot
            + kv_head_idx * kc_stride_head
            + offs_d[:, None] * kc_stride_dim
        )
        key = tl.load(
            key_cache_ptr + k_offsets,
            mask=dim_mask[:, None] & token_valid[None, :],
            other=0.0,
        )

        if USE_SMALL_K and BLOCK_N == 1:
            key_vec = tl.sum(key.to(tl.float32), axis=1)
            score_vec = tl.sum(
                query.to(tl.float32) * tl.expand_dims(key_vec, 0),
                axis=1,
            ) * sm_scale
            scores = tl.expand_dims(score_vec, 1)
        else:
            scores = tl.dot(query, key) * sm_scale
        attn_mask = (
            q_mask[:, None]
            & token_valid[None, :]
            & (q_abs[:, None] >= k_abs[None, :])
        )
        masked_scores = tl.where(attn_mask, scores, float("-inf"))
        row_has_valid = tl.max(attn_mask.to(tl.int32), axis=1) > 0
        block_max = tl.max(masked_scores, axis=1)
        safe_block_max = tl.where(row_has_valid, block_max, 0.0)
        new_max = tl.where(
            row_has_valid,
            tl.maximum(running_max, block_max),
            running_max,
        )
        # Avoid exp2(-inf - -inf) before a row has seen any valid block.
        prev_scale = tl.where(
            running_max == float("-inf"),
            0.0,
            tl.math.exp2(running_max - new_max),
        )
        block_scale = tl.where(
            row_has_valid,
            tl.math.exp2(safe_block_max - new_max),
            0.0,
        )
        probs = tl.where(
            attn_mask,
            tl.math.exp2(masked_scores - safe_block_max[:, None]),
            0.0,
        )

        v_offsets = (
            physical_block_idx[:, None] * vc_stride_blk
            + slot[:, None] * vc_stride_slot
            + kv_head_idx * vc_stride_head
            + offs_d[None, :] * vc_stride_dim
        )
        value = tl.load(
            value_cache_ptr + v_offsets,
            mask=token_valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        if USE_SMALL_K:
            value_acc = _smallk_weighted_value_sum(
                probs,
                value,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
            )
        else:
            value_acc = tl.dot(probs.to(value.dtype), value)
        running_out = (
            running_out * prev_scale[:, None]
            + value_acc * block_scale[:, None]
        )
        running_denom = (
            running_denom * prev_scale
            + tl.sum(probs, axis=1) * block_scale
        )
        running_max = new_max

    output = running_out / tl.maximum(running_denom[:, None], 1e-20)
    out_offsets = (
        safe_q_pos[:, None] * out_stride_tok
        + head_idx * out_stride_head
        + offs_d[None, :] * out_stride_dim
    )
    tl.store(
        output_ptr + out_offsets,
        output,
        mask=q_mask[:, None] & dim_mask[None, :],
    )


def _launch_contiguous_sparse_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_metadata: SparseTopKBlockMetadata,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    output_dtype: torch.dtype | None,
    breakdown: SparsePrefillTimingBreakdown | None = None,
) -> torch.Tensor:
    output = torch.empty_like(
        query,
        dtype=query.dtype if output_dtype is None else output_dtype,
    )
    block_m = triton.next_power_of_2(int(cfg.q_block))
    block_n = triton.next_power_of_2(int(cfg.k_block))
    use_small_k = block_n < 16
    head_size = int(query.shape[-1])
    head_size_padded = triton.next_power_of_2(head_size)
    num_queries_per_kv = query.shape[1] // key.shape[1]
    grid = (topk_metadata.q_blocks, query.shape[1])
    num_warps = 4 if head_size <= 64 else 8
    _timed_call(
        breakdown,
        "attn_kernel",
        lambda: _sparse_prefill_attention_contiguous_kernel[grid](
            query,
            key,
            value,
            topk_metadata.topk_block_indices,
            topk_metadata.topk_block_counts,
            output,
            float(scaling) * RCP_LN2,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            topk_metadata.topk_block_indices.stride(0),
            topk_metadata.topk_block_indices.stride(1),
            topk_metadata.topk_block_indices.stride(2),
            topk_metadata.topk_block_counts.stride(0),
            topk_metadata.topk_block_counts.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            query.shape[0],
            key.shape[0],
            topk_metadata.q_abs_offset,
            num_queries_per_kv,
            Q_BLOCK=int(cfg.q_block),
            K_BLOCK=int(cfg.k_block),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            TOPK=int(topk_metadata.topk_block_indices.shape[2]),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_SMALL_K=use_small_k,
            num_warps=num_warps,
            num_stages=1,
        ),
    )
    if int(cfg.k_block) == 1:
        value_mean = _timed_call(
            breakdown,
            "attn_k1_value_mean",
            lambda: value.to(torch.float32).mean(dim=0),
        )
        output = _timed_call(
            breakdown,
            "attn_k1_reference_fallback",
            _apply_k1_reference_fallback,
            output=output,
            value_mean=value_mean,
            topk_metadata=topk_metadata,
            q_block=int(cfg.q_block),
        )
    return output


def _launch_paged_sparse_attention(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    kv_cache_dtype: str,
    topk_metadata: SparseTopKBlockMetadata,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    output_dtype: torch.dtype | None,
    breakdown: SparsePrefillTimingBreakdown | None = None,
) -> torch.Tensor:
    key_cache, value_cache = _timed_call(
        breakdown,
        "attn_extract_kv",
        _extract_unquantized_paged_kv,
        kv_cache,
        kv_cache_dtype=kv_cache_dtype,
    )
    cache_block_size = int(key_cache.shape[1])
    output = torch.empty_like(
        query,
        dtype=query.dtype if output_dtype is None else output_dtype,
    )
    block_table_row = _timed_call(
        breakdown,
        "attn_prepare_block_table",
        lambda: block_table_row.to(device=query.device, dtype=torch.int32).contiguous(),
    )
    block_m = triton.next_power_of_2(int(cfg.q_block))
    block_n = triton.next_power_of_2(int(cfg.k_block))
    use_small_k = block_n < 16
    head_size = int(query.shape[-1])
    head_size_padded = triton.next_power_of_2(head_size)
    num_queries_per_kv = query.shape[1] // key_cache.shape[2]
    grid = (topk_metadata.q_blocks, query.shape[1])
    num_warps = 4 if head_size <= 64 else 8
    _timed_call(
        breakdown,
        "attn_kernel",
        lambda: _sparse_prefill_attention_paged_kernel[grid](
            query,
            key_cache,
            value_cache,
            block_table_row,
            topk_metadata.topk_block_indices,
            topk_metadata.topk_block_counts,
            output,
            float(scaling) * RCP_LN2,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            topk_metadata.topk_block_indices.stride(0),
            topk_metadata.topk_block_indices.stride(1),
            topk_metadata.topk_block_indices.stride(2),
            topk_metadata.topk_block_counts.stride(0),
            topk_metadata.topk_block_counts.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            query.shape[0],
            seq_len,
            topk_metadata.q_abs_offset,
            num_queries_per_kv,
            CACHE_BLOCK_SIZE=cache_block_size,
            Q_BLOCK=int(cfg.q_block),
            K_BLOCK=int(cfg.k_block),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            TOPK=int(topk_metadata.topk_block_indices.shape[2]),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_SMALL_K=use_small_k,
            num_warps=num_warps,
            num_stages=1,
        ),
    )
    if int(cfg.k_block) == 1:
        value_mean = _timed_call(
            breakdown,
            "attn_k1_value_mean",
            _gather_paged_value_mean,
            value_cache=value_cache,
            block_table_row=block_table_row,
            seq_len=seq_len,
        )
        output = _timed_call(
            breakdown,
            "attn_k1_reference_fallback",
            _apply_k1_reference_fallback,
            output=output,
            value_mean=value_mean,
            topk_metadata=topk_metadata,
            q_block=int(cfg.q_block),
        )
    return output


@triton.jit
def _retain_score_paged_kernel(
    query_ptr,
    key_cache_ptr,
    block_table_ptr,
    retain_block_mask_ptr,
    output_ptr,
    sm_scale,
    q_stride_tok,
    q_stride_head,
    q_stride_dim,
    kc_stride_blk,
    kc_stride_slot,
    kc_stride_head,
    kc_stride_dim,
    retain_mask_head_stride,
    retain_mask_qblock_stride,
    retain_mask_kiter_stride,
    retain_mask_word_stride,
    out_stride_tok,
    out_stride_head,
    q_len,
    seq_len,
    q_abs_offset,
    num_queries_per_kv,
    num_k_iters,
    K_BLOCK_SPARSE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_queries_per_kv

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    q_pos = q_block_idx * Q_BLOCK + offs_m
    q_mask = (offs_m < Q_BLOCK) & (q_pos < q_len)
    safe_q_pos = tl.where(q_mask, q_pos, 0)
    q_abs = safe_q_pos + q_abs_offset
    dim_mask = offs_d < HEAD_SIZE

    q_offsets = (
        safe_q_pos[:, None] * q_stride_tok
        + head_idx * q_stride_head
        + offs_d[None, :] * q_stride_dim
    )
    query = tl.load(
        query_ptr + q_offsets,
        mask=q_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    running_denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    running_selected = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_iter in range(num_k_iters):
        offs_n = tl.arange(0, BLOCK_N)
        k_abs = k_iter * BLOCK_N + offs_n
        token_valid = k_abs < seq_len
        safe_k_abs = tl.where(token_valid, k_abs, 0)

        physical_block_idx = tl.load(
            block_table_ptr + (safe_k_abs // CACHE_BLOCK_SIZE),
            mask=token_valid,
            other=0,
        )
        slot = safe_k_abs % CACHE_BLOCK_SIZE
        k_offsets = (
            physical_block_idx[None, :] * kc_stride_blk
            + slot[None, :] * kc_stride_slot
            + kv_head_idx * kc_stride_head
            + offs_d[:, None] * kc_stride_dim
        )
        key = tl.load(
            key_cache_ptr + k_offsets,
            mask=dim_mask[:, None] & token_valid[None, :],
            other=0.0,
        )

        scores = tl.dot(query, key) * sm_scale

        attn_mask = (
            q_mask[:, None]
            & token_valid[None, :]
            & (q_abs[:, None] >= k_abs[None, :])
        )
        masked_scores = tl.where(attn_mask, scores, float("-inf"))

        row_has_valid = tl.max(attn_mask.to(tl.int32), axis=1) > 0
        block_max = tl.max(masked_scores, axis=1)
        safe_block_max = tl.where(row_has_valid, block_max, 0.0)
        new_max = tl.where(
            row_has_valid,
            tl.maximum(running_max, block_max),
            running_max,
        )
        prev_scale = tl.where(
            running_max == float("-inf"),
            0.0,
            tl.math.exp2(running_max - new_max),
        )
        block_scale = tl.where(
            row_has_valid,
            tl.math.exp2(safe_block_max - new_max),
            0.0,
        )
        probs = tl.where(
            attn_mask,
            tl.math.exp2(masked_scores - safe_block_max[:, None]),
            0.0,
        )

        block_denom = tl.sum(probs, axis=1)
        running_denom = running_denom * prev_scale + block_denom * block_scale

        k_block_sparse = tl.where(token_valid, safe_k_abs // K_BLOCK_SPARSE, 0)
        tile_sparse_base = (k_iter * BLOCK_N) // K_BLOCK_SPARSE
        local_block_idx = tl.where(token_valid, k_block_sparse - tile_sparse_base, 0)
        local_word_idx = local_block_idx // 32
        local_bit_idx = local_block_idx % 32
        bit_mask = 1 << local_bit_idx
        retain_mask_base = (
            head_idx * retain_mask_head_stride
            + q_block_idx * retain_mask_qblock_stride
            + k_iter * retain_mask_kiter_stride
        )
        is_selected = tl.zeros([BLOCK_N], dtype=tl.int32)
        for word_idx in tl.static_range(0, 4):
            word_value = tl.load(
                retain_block_mask_ptr
                + retain_mask_base
                + word_idx * retain_mask_word_stride
            )
            word_match = token_valid & (local_word_idx == word_idx)
            has_bit = (word_value & bit_mask) != 0
            is_selected = tl.where(word_match & has_bit, 1, is_selected)

        selected_probs = tl.where(
            (is_selected > 0)[None, :] & attn_mask,
            probs,
            0.0,
        )
        selected_denom = tl.sum(selected_probs, axis=1)
        running_selected = running_selected * prev_scale + selected_denom * block_scale

        running_max = new_max

    retain_score = running_selected / tl.maximum(running_denom, 1e-20)
    retain_score = tl.where(q_mask, retain_score, 0.0)

    out_offsets = safe_q_pos * out_stride_tok + head_idx * out_stride_head
    tl.store(output_ptr + out_offsets, retain_score, mask=q_mask)


def _compute_paged_retain_scores(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    kv_cache_dtype: str,
    topk_metadata: SparseTopKBlockMetadata,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    breakdown: SparsePrefillTimingBreakdown | None = None,
) -> torch.Tensor:
    """Compute per-token retain scores for paged KV. Returns [q_len, num_heads] float32."""
    retain_block_mask = topk_metadata.retain_tile_block_mask
    if retain_block_mask is None:
        raise ValueError(
            "Paged retain-score kernel requires precomputed retain_tile_block_mask."
        )
    key_cache, _ = _timed_call(
        breakdown,
        "retain_extract_kv",
        _extract_unquantized_paged_kv,
        kv_cache,
        kv_cache_dtype=kv_cache_dtype,
    )
    cache_block_size = int(key_cache.shape[1])
    num_heads = query.shape[1]
    q_len_val = int(query.shape[0])
    output = torch.empty(q_len_val, num_heads, device=query.device, dtype=torch.float32)

    block_table_int = _timed_call(
        breakdown,
        "retain_prepare_block_table",
        lambda: block_table_row.to(device=query.device, dtype=torch.int32).contiguous(),
    )
    block_m = max(triton.next_power_of_2(int(cfg.q_block)), 16)
    head_size = int(query.shape[-1])
    head_size_padded = triton.next_power_of_2(head_size)
    num_queries_per_kv = num_heads // key_cache.shape[2]

    BLOCK_N = RETAIN_TILE_SIZE
    num_k_iters = _ceil_div(seq_len, BLOCK_N)

    grid = (topk_metadata.q_blocks, num_heads)
    num_warps = 4 if head_size <= 64 else 8

    _timed_call(
        breakdown,
        "retain_kernel",
        lambda: _retain_score_paged_kernel[grid](
            query,
            key_cache,
            block_table_int,
            retain_block_mask,
            output,
            float(scaling) * RCP_LN2,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            retain_block_mask.stride(0),
            retain_block_mask.stride(1),
            retain_block_mask.stride(2),
            retain_block_mask.stride(3),
            output.stride(0),
            output.stride(1),
            q_len_val,
            seq_len,
            topk_metadata.q_abs_offset,
            num_queries_per_kv,
            num_k_iters,
            K_BLOCK_SPARSE=int(cfg.k_block),
            CACHE_BLOCK_SIZE=cache_block_size,
            Q_BLOCK=int(cfg.q_block),
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            num_warps=num_warps,
            num_stages=1,
        ),
    )
    return output


def _build_token_level_retain_stats(
    per_token_retain_scores: torch.Tensor,
    existing_stats: SparsePrefillSelectionStats,
) -> SparsePrefillSelectionStats:
    """Build selection stats with token-level retain scores, keeping block-level density."""
    q_len = per_token_retain_scores.shape[0]
    num_heads = per_token_retain_scores.shape[1]
    per_head_retain_sum = per_token_retain_scores.to(torch.float64).sum(dim=0)
    return SparsePrefillSelectionStats(
        total_valid_blocks=existing_stats.total_valid_blocks,
        total_kept_blocks=existing_stats.total_kept_blocks,
        total_valid_rows=q_len * num_heads,
        retained_attention_score_sum=float(per_head_retain_sum.sum().item()),
        per_head_total_valid_blocks=existing_stats.per_head_total_valid_blocks,
        per_head_total_kept_blocks=existing_stats.per_head_total_kept_blocks,
        per_head_total_valid_rows=tuple(q_len for _ in range(num_heads)),
        per_head_retained_attention_score_sum=tuple(
            float(v) for v in per_head_retain_sum.tolist()
        ),
    )


def _log_triton_retain_score_stats(
    *,
    layer: object | None,
    query_len: int,
    seq_len: int,
    cfg: SparsePrefillTopKConfig,
    paged_kv: bool,
    selection_stats: SparsePrefillSelectionStats | None,
    retain_score_log_mode: SparsePrefillRetainScoreLogMode,
) -> None:
    if selection_stats is None:
        return
    layer_idx = None
    layer_name = None
    selection_policy = format_sparse_prefill_selection_policy(cfg)
    if should_log_sparse_prefill_layer_info(retain_score_log_mode):
        layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)
        logger.info(
            "Sparse prefill Triton retain-score stats: layer_idx=%s "
            "layer_name=%s q_len=%d seq_len=%d path=%s q_block=%d k_block=%d "
            "%s avg_retain_score=%.6f density=%.6f valid_rows=%d "
            "kept_blocks=%d valid_blocks=%d.",
            layer_idx if layer_idx is not None else "NA",
            layer_name or "unknown",
            int(query_len),
            int(seq_len),
            "paged" if paged_kv else "contiguous",
            int(cfg.q_block),
            int(cfg.k_block),
            selection_policy,
            selection_stats.retained_attention_score_mean,
            selection_stats.density,
            selection_stats.total_valid_rows,
            selection_stats.total_kept_blocks,
            selection_stats.total_valid_blocks,
        )
    else:
        logger.info(
            "Sparse prefill Triton retain-score stats: q_len=%d seq_len=%d "
            "path=%s q_block=%d k_block=%d %s avg_retain_score=%.6f "
            "density=%.6f valid_rows=%d kept_blocks=%d valid_blocks=%d.",
            int(query_len),
            int(seq_len),
            "paged" if paged_kv else "contiguous",
            int(cfg.q_block),
            int(cfg.k_block),
            selection_policy,
            selection_stats.retained_attention_score_mean,
            selection_stats.density,
            selection_stats.total_valid_rows,
            selection_stats.total_kept_blocks,
            selection_stats.total_valid_blocks,
        )
    if should_log_sparse_prefill_per_head_stats(retain_score_log_mode):
        per_head_payload = format_sparse_prefill_per_head_payload(selection_stats)
        logger.info(
            "Sparse prefill Triton retain-score per-head stats: layer_idx=%s "
            "layer_name=%s q_len=%d seq_len=%d path=%s q_block=%d k_block=%d "
            "%s num_heads=%d retain_scores=%s densities=%s valid_rows=%s "
            "kept_blocks=%s valid_blocks=%s.",
            layer_idx if layer_idx is not None else "NA",
            layer_name or "unknown",
            int(query_len),
            int(seq_len),
            "paged" if paged_kv else "contiguous",
            int(cfg.q_block),
            int(cfg.k_block),
            selection_policy,
            len(selection_stats.per_head_total_valid_rows),
            per_head_payload["retain_scores"],
            per_head_payload["densities"],
            per_head_payload["valid_rows"],
            per_head_payload["kept_blocks"],
            per_head_payload["valid_blocks"],
        )


def run_triton_sparse_prefill_attention(
    *,
    query: torch.Tensor,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    output_dtype: torch.dtype | None = None,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    block_table_row: torch.Tensor | None = None,
    seq_len: int | None = None,
    kv_cache_dtype: str = "auto",
    record_retain_score: bool = False,
    retain_score_log_mode: SparsePrefillRetainScoreLogMode | None = None,
    layer: object | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available for sparse prefill attention.")
    if query.device.type != "cuda":
        raise RuntimeError("Triton sparse prefill attention requires CUDA tensors.")
    timing_breakdown = SparsePrefillTimingBreakdown(
        enabled=_is_sparse_prefill_timing_breakdown_enabled(),
        device=query.device,
        timings_ms={},
    )
    if timing_breakdown.enabled:
        torch.cuda.synchronize(device=query.device)
        total_start = time.perf_counter()
    else:
        total_start = 0.0
    resolved_log_mode = resolve_sparse_prefill_retain_score_log_mode(
        record_retain_score=record_retain_score,
        retain_score_log_mode=retain_score_log_mode,
    )

    if kv_cache is not None and block_table_row is not None and seq_len is not None:
        topk_metadata = build_sparse_topk_block_metadata_from_paged_cache(
            query=query,
            kv_cache=kv_cache,
            block_table_row=block_table_row,
            seq_len=seq_len,
            scaling=scaling,
            kv_cache_dtype=kv_cache_dtype,
            cfg=cfg,
            build_retain_tile_block_mask=resolved_log_mode != "off",
            record_selection_stats=resolved_log_mode != "off",
            breakdown=timing_breakdown,
        )
        selection_stats_for_log = topk_metadata.selection_stats
        should_compute_token_retain = (
            resolved_log_mode != "off"
            and topk_metadata.selection_stats is not None
            and topk_metadata.selection_stats.total_kept_blocks
            < topk_metadata.selection_stats.total_valid_blocks
        )
        if (
            should_compute_token_retain
            and topk_metadata.selection_stats is not None
        ):
            per_token_scores = _compute_paged_retain_scores(
                query=query,
                kv_cache=kv_cache,
                block_table_row=block_table_row,
                seq_len=seq_len,
                kv_cache_dtype=kv_cache_dtype,
                topk_metadata=topk_metadata,
                scaling=scaling,
                cfg=cfg,
                breakdown=timing_breakdown,
            )
            selection_stats_for_log = _build_token_level_retain_stats(
                per_token_scores, topk_metadata.selection_stats,
            )
        _log_triton_retain_score_stats(
            layer=layer,
            query_len=int(query.shape[0]),
            seq_len=int(seq_len),
            cfg=cfg,
            paged_kv=True,
            selection_stats=selection_stats_for_log,
            retain_score_log_mode=resolved_log_mode,
        )
        _log_fully_masked_row_debug(
            topk_metadata=topk_metadata,
            q_len=int(query.shape[0]),
            seq_len=int(seq_len),
            cfg=cfg,
            paged_kv=True,
            layer=layer,
        )
        output = _launch_paged_sparse_attention(
            query=query,
            kv_cache=kv_cache,
            block_table_row=block_table_row,
            seq_len=seq_len,
            kv_cache_dtype=kv_cache_dtype,
            topk_metadata=topk_metadata,
            scaling=scaling,
            cfg=cfg,
            output_dtype=output_dtype,
            breakdown=timing_breakdown,
        )
        if timing_breakdown.enabled:
            torch.cuda.synchronize(device=query.device)
            _log_sparse_prefill_timing_breakdown(
                layer=layer,
                query_len=int(query.shape[0]),
                seq_len=int(seq_len),
                paged_kv=True,
                total_ms=(time.perf_counter() - total_start) * 1000.0,
                breakdown=timing_breakdown,
            )
        return output

    if key is None or value is None:
        raise ValueError(
            "Sparse Triton prefill requires either (key, value) or "
            "(kv_cache, block_table_row, seq_len)."
        )

    pooled_key = _timed_call(
        timing_breakdown,
        "contiguous_pool_key",
        _mean_pool_attention_blocks,
        key.transpose(0, 1).unsqueeze(0),
        cfg.k_block,
    )
    topk_metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=scaling,
        cfg=cfg,
        k_len=key.shape[0],
        record_selection_stats=resolved_log_mode != "off",
        breakdown=timing_breakdown,
    )
    if (
        resolved_log_mode != "off"
        and topk_metadata.selection_stats is not None
        and topk_metadata.selection_stats.total_kept_blocks
        < topk_metadata.selection_stats.total_valid_blocks
    ):
        logger.info(
            "contiguous retain score not computed "
            "(kept_blocks=%d < valid_blocks=%d)",
            topk_metadata.selection_stats.total_kept_blocks,
            topk_metadata.selection_stats.total_valid_blocks,
        )
    _log_triton_retain_score_stats(
        layer=layer,
        query_len=int(query.shape[0]),
        seq_len=int(key.shape[0]),
        cfg=cfg,
        paged_kv=False,
        selection_stats=topk_metadata.selection_stats,
        retain_score_log_mode=resolved_log_mode,
    )
    _log_fully_masked_row_debug(
        topk_metadata=topk_metadata,
        q_len=int(query.shape[0]),
        seq_len=int(key.shape[0]),
        cfg=cfg,
        paged_kv=False,
        layer=layer,
    )
    output = _launch_contiguous_sparse_attention(
        query=query,
        key=key,
        value=value,
        topk_metadata=topk_metadata,
        scaling=scaling,
        cfg=cfg,
        output_dtype=output_dtype,
        breakdown=timing_breakdown,
    )
    if timing_breakdown.enabled:
        torch.cuda.synchronize(device=query.device)
        _log_sparse_prefill_timing_breakdown(
            layer=layer,
            query_len=int(query.shape[0]),
            seq_len=int(key.shape[0]),
            paged_kv=False,
            total_ms=(time.perf_counter() - total_start) * 1000.0,
            breakdown=timing_breakdown,
        )
    return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import RCP_LN2
from vllm.v1.attention.backends.sparse_prefill_utils import (
    SparsePrefillRetainScoreLogMode,
    SparsePrefillSelectionStats,
    SparsePrefillTopKConfig,
    _mean_pool_attention_blocks,
    build_sparse_prefill_selection_stats,
    format_sparse_prefill_per_head_payload,
    resolve_sparse_prefill_layer_info,
    resolve_sparse_prefill_retain_score_log_mode,
    should_log_sparse_prefill_layer_info,
    should_log_sparse_prefill_per_head_stats,
)

logger = init_logger(__name__)


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


def _summarize_topk_block_selection(
    *,
    masked_scores: torch.Tensor,
    topk_values: torch.Tensor,
    valid_block_mask: torch.Tensor,
    counts: torch.Tensor,
) -> SparsePrefillSelectionStats:
    valid_block_counts = valid_block_mask.sum(dim=-1).to(torch.int32)
    valid_row_mask = valid_block_counts > 0
    safe_logsumexp = torch.logsumexp(masked_scores, dim=-1)
    safe_logsumexp = torch.where(
        valid_row_mask,
        safe_logsumexp,
        torch.zeros_like(safe_logsumexp),
    )

    rank = torch.arange(
        topk_values.shape[-1],
        device=topk_values.device,
        dtype=counts.dtype,
    ).view(*([1] * counts.dim()), topk_values.shape[-1])
    keep_mask = rank < counts.unsqueeze(-1)
    retained_attention_score_mass = torch.where(
        keep_mask & valid_row_mask.unsqueeze(-1),
        torch.exp(topk_values - safe_logsumexp.unsqueeze(-1)),
        torch.zeros_like(topk_values),
    ).sum(dim=-1)

    return build_sparse_prefill_selection_stats(
        valid_block_counts=valid_block_counts,
        kept_block_counts=counts,
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
    record_selection_stats: bool = False,
) -> SparseTopKBlockMetadata:
    query_states = query.transpose(0, 1).unsqueeze(0)
    pooled_query = _mean_pool_attention_blocks(query_states, cfg.q_block)
    if pooled_key.shape[1] != pooled_query.shape[1]:
        if pooled_query.shape[1] % pooled_key.shape[1] != 0:
            raise ValueError(
                "Sparse Triton prefill requires query heads to be divisible "
                f"by KV heads, got num_heads={pooled_query.shape[1]} and "
                f"num_kv_heads={pooled_key.shape[1]}."
            )
        pooled_key = _repeat_kv_heads(
            pooled_key, pooled_query.shape[1] // pooled_key.shape[1]
        )

    block_scores = torch.matmul(
        pooled_query.to(torch.float32),
        pooled_key.to(torch.float32).transpose(-1, -2),
    ) * float(scaling)
    valid_block_mask = _build_causal_valid_block_mask(
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
    masked_scores = torch.where(
        valid_block_mask,
        block_scores,
        torch.full_like(block_scores, float("-inf")),
    )
    k_keep = min(int(cfg.topk), int(masked_scores.shape[-1]))
    topk_result = torch.topk(masked_scores, k=k_keep, dim=-1)
    topk_idx = topk_result.indices
    counts = valid_block_mask.sum(dim=-1).clamp_max(k_keep).to(torch.int32)
    padded_topk_idx = _pad_topk_indices(topk_idx, counts, full_topk=int(cfg.topk))
    selection_stats = (
        _summarize_topk_block_selection(
            masked_scores=masked_scores,
            topk_values=topk_result.values,
            valid_block_mask=valid_block_mask,
            counts=counts,
        )
        if record_selection_stats
        else None
    )
    return SparseTopKBlockMetadata(
        topk_block_indices=padded_topk_idx.squeeze(0).contiguous(),
        topk_block_counts=counts.squeeze(0).contiguous(),
        q_blocks=_ceil_div(query.shape[0], cfg.q_block),
        k_blocks=_ceil_div(k_len, cfg.k_block),
        q_abs_offset=k_len - query.shape[0],
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
    record_selection_stats: bool = False,
) -> SparseTopKBlockMetadata:
    key_cache, _ = _extract_unquantized_paged_kv(
        kv_cache,
        kv_cache_dtype=kv_cache_dtype,
    )
    pooled_key = _pool_paged_key_blocks(
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
        record_selection_stats=record_selection_stats,
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


def _fully_masked_row_mask_for_k1(
    *,
    topk_metadata: SparseTopKBlockMetadata,
    q_len: int,
    q_block: int,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(q_len, device=device, dtype=torch.int64)
    q_block_idx = torch.div(q_positions, q_block, rounding_mode="floor")
    selected = topk_metadata.topk_block_indices.to(device=device, dtype=torch.int64).index_select(
        1,
        q_block_idx,
    )
    q_abs = q_positions + int(topk_metadata.q_abs_offset)
    has_valid = (
        (selected >= 0)
        & (selected <= q_abs.view(1, q_len, 1))
    ).any(dim=-1)
    return (~has_valid).transpose(0, 1).contiguous()


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
    _sparse_prefill_attention_contiguous_kernel[grid](
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
        TOPK=int(cfg.topk),
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_SMALL_K=use_small_k,
        num_warps=num_warps,
        num_stages=1,
    )
    if int(cfg.k_block) == 1:
        output = _apply_k1_reference_fallback(
            output=output,
            value_mean=value.to(torch.float32).mean(dim=0),
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
) -> torch.Tensor:
    key_cache, value_cache = _extract_unquantized_paged_kv(
        kv_cache,
        kv_cache_dtype=kv_cache_dtype,
    )
    cache_block_size = int(key_cache.shape[1])
    output = torch.empty_like(
        query,
        dtype=query.dtype if output_dtype is None else output_dtype,
    )
    block_table_row = block_table_row.to(device=query.device, dtype=torch.int32).contiguous()
    block_m = triton.next_power_of_2(int(cfg.q_block))
    block_n = triton.next_power_of_2(int(cfg.k_block))
    use_small_k = block_n < 16
    head_size = int(query.shape[-1])
    head_size_padded = triton.next_power_of_2(head_size)
    num_queries_per_kv = query.shape[1] // key_cache.shape[2]
    grid = (topk_metadata.q_blocks, query.shape[1])
    num_warps = 4 if head_size <= 64 else 8
    _sparse_prefill_attention_paged_kernel[grid](
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
        TOPK=int(cfg.topk),
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_SMALL_K=use_small_k,
        num_warps=num_warps,
        num_stages=1,
    )
    if int(cfg.k_block) == 1:
        output = _apply_k1_reference_fallback(
            output=output,
            value_mean=_gather_paged_value_mean(
                value_cache=value_cache,
                block_table_row=block_table_row,
                seq_len=seq_len,
            ),
            topk_metadata=topk_metadata,
            q_block=int(cfg.q_block),
        )
    return output


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
    if should_log_sparse_prefill_layer_info(retain_score_log_mode):
        layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)
        logger.info(
            "Sparse prefill Triton retain-score stats: layer_idx=%s "
            "layer_name=%s q_len=%d seq_len=%d path=%s q_block=%d k_block=%d "
            "topk=%d avg_retain_score=%.6f density=%.6f valid_rows=%d "
            "kept_blocks=%d valid_blocks=%d.",
            layer_idx if layer_idx is not None else "NA",
            layer_name or "unknown",
            int(query_len),
            int(seq_len),
            "paged" if paged_kv else "contiguous",
            int(cfg.q_block),
            int(cfg.k_block),
            int(cfg.topk),
            selection_stats.retained_attention_score_mean,
            selection_stats.density,
            selection_stats.total_valid_rows,
            selection_stats.total_kept_blocks,
            selection_stats.total_valid_blocks,
        )
    else:
        logger.info(
            "Sparse prefill Triton retain-score stats: q_len=%d seq_len=%d "
            "path=%s q_block=%d k_block=%d topk=%d avg_retain_score=%.6f "
            "density=%.6f valid_rows=%d kept_blocks=%d valid_blocks=%d.",
            int(query_len),
            int(seq_len),
            "paged" if paged_kv else "contiguous",
            int(cfg.q_block),
            int(cfg.k_block),
            int(cfg.topk),
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
            "topk=%d num_heads=%d retain_scores=%s densities=%s valid_rows=%s "
            "kept_blocks=%s valid_blocks=%s.",
            layer_idx if layer_idx is not None else "NA",
            layer_name or "unknown",
            int(query_len),
            int(seq_len),
            "paged" if paged_kv else "contiguous",
            int(cfg.q_block),
            int(cfg.k_block),
            int(cfg.topk),
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
            record_selection_stats=resolved_log_mode != "off",
        )
        _log_triton_retain_score_stats(
            layer=layer,
            query_len=int(query.shape[0]),
            seq_len=int(seq_len),
            cfg=cfg,
            paged_kv=True,
            selection_stats=topk_metadata.selection_stats,
            retain_score_log_mode=resolved_log_mode,
        )
        return _launch_paged_sparse_attention(
            query=query,
            kv_cache=kv_cache,
            block_table_row=block_table_row,
            seq_len=seq_len,
            kv_cache_dtype=kv_cache_dtype,
            topk_metadata=topk_metadata,
            scaling=scaling,
            cfg=cfg,
            output_dtype=output_dtype,
        )

    if key is None or value is None:
        raise ValueError(
            "Sparse Triton prefill requires either (key, value) or "
            "(kv_cache, block_table_row, seq_len)."
        )

    pooled_key = _mean_pool_attention_blocks(
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
    return _launch_contiguous_sparse_attention(
        query=query,
        key=key,
        value=value,
        topk_metadata=topk_metadata,
        scaling=scaling,
        cfg=cfg,
        output_dtype=output_dtype,
    )

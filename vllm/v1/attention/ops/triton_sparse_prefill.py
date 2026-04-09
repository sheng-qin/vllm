# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import RCP_LN2
from vllm.v1.attention.backends.sparse_prefill_utils import (
    SparsePrefillTopKConfig,
    _mean_pool_attention_blocks,
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


def build_sparse_topk_block_metadata(
    *,
    query: torch.Tensor,
    pooled_key: torch.Tensor,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    k_len: int,
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
    topk_idx = torch.topk(masked_scores, k=k_keep, dim=-1).indices
    counts = valid_block_mask.sum(dim=-1).clamp_max(k_keep).to(torch.int32)
    padded_topk_idx = _pad_topk_indices(topk_idx, counts, full_topk=int(cfg.topk))
    return SparseTopKBlockMetadata(
        topk_block_indices=padded_topk_idx.squeeze(0).contiguous(),
        topk_block_counts=counts.squeeze(0).contiguous(),
        q_blocks=_ceil_div(query.shape[0], cfg.q_block),
        k_blocks=_ceil_div(k_len, cfg.k_block),
        q_abs_offset=k_len - query.shape[0],
    )


def build_sparse_topk_block_metadata_from_paged_cache(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    kv_cache_dtype: str,
    cfg: SparsePrefillTopKConfig,
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
        scaling=1.0,
        cfg=cfg,
        k_len=seq_len,
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

        running_out = (
            running_out * prev_scale[:, None]
            + tl.dot(probs.to(value.dtype), value) * block_scale[:, None]
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

        running_out = (
            running_out * prev_scale[:, None]
            + tl.dot(probs.to(value.dtype), value) * block_scale[:, None]
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
        num_warps=num_warps,
        num_stages=1,
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
        num_warps=num_warps,
        num_stages=1,
    )
    return output


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
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available for sparse prefill attention.")
    if query.device.type != "cuda":
        raise RuntimeError("Triton sparse prefill attention requires CUDA tensors.")

    if kv_cache is not None and block_table_row is not None and seq_len is not None:
        topk_metadata = build_sparse_topk_block_metadata_from_paged_cache(
            query=query,
            kv_cache=kv_cache,
            block_table_row=block_table_row,
            seq_len=seq_len,
            kv_cache_dtype=kv_cache_dtype,
            cfg=cfg,
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

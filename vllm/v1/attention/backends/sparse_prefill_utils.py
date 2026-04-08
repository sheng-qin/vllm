# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for env-gated sparse prefill attention prototypes."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

AUTOPTQ_VLLM_SPARSE_ENABLE_ENV = "AUTOPTQ_VLLM_SPARSE_ENABLE"
AUTOPTQ_SPARSE_RUNTIME_JSON_ENV = "AUTOPTQ_SPARSE_RUNTIME_JSON"
AUTOPTQ_SPARSE_RUNTIME_KEY_ENV = "AUTOPTQ_SPARSE_RUNTIME_KEY"

DEFAULT_PV_BLOCK_SIZE = 128


@dataclass(frozen=True)
class SparsePrefillTopKConfig:
    key: str
    name: str
    q_block: int
    k_block: int
    topk: int
    pv_block_size: int = DEFAULT_PV_BLOCK_SIZE


def _parse_env_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_sparse_prefill_enabled() -> bool:
    return _parse_env_flag(os.getenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV))


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Sparse scheme field '{field_name}' must be an integer.")
    if value <= 0:
        raise ValueError(f"Sparse scheme field '{field_name}' must be > 0.")
    return int(value)


@lru_cache(maxsize=None)
def _load_sparse_prefill_topk_config(
    json_path: str, key: str
) -> SparsePrefillTopKConfig:
    try:
        with open(json_path) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(
            f"Failed to read sparse scheme JSON from '{json_path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Sparse scheme JSON '{json_path}' must contain a top-level object."
        )
    if key not in payload:
        raise ValueError(
            f"Sparse scheme key '{key}' was not found in '{json_path}'."
        )

    scheme = payload[key]
    if not isinstance(scheme, dict):
        raise ValueError(
            f"Sparse scheme '{key}' in '{json_path}' must be a JSON object."
        )

    if "topk" not in scheme or scheme.get("topk") is None:
        raise ValueError(
            f"Only topk sparse schemes are supported in vLLM v1. "
            f"Scheme '{key}' must define 'topk'."
        )
    if scheme.get("ratio") is not None or scheme.get("threshold") is not None:
        raise ValueError(
            f"Only topk sparse schemes are supported in vLLM v1. "
            f"Scheme '{key}' must not define 'ratio' or 'threshold'."
        )

    return SparsePrefillTopKConfig(
        key=key,
        name=str(scheme.get("name", key)),
        q_block=_require_positive_int(scheme.get("q_block"), "q_block"),
        k_block=_require_positive_int(scheme.get("k_block"), "k_block"),
        topk=_require_positive_int(scheme.get("topk"), "topk"),
    )


def get_sparse_prefill_topk_config() -> SparsePrefillTopKConfig | None:
    if not is_sparse_prefill_enabled():
        return None

    json_path = os.getenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV)
    key = os.getenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV)
    if not json_path or not key:
        raise ValueError(
            "Sparse prefill backend requires both "
            f"{AUTOPTQ_SPARSE_RUNTIME_JSON_ENV} and "
            f"{AUTOPTQ_SPARSE_RUNTIME_KEY_ENV} to be set."
        )
    return _load_sparse_prefill_topk_config(json_path=json_path, key=key)


def is_full_prefill_request(query_len: int, seq_len: int) -> bool:
    if query_len <= 1:
        return False
    return query_len == seq_len


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    )


def _broadcast_attention_mask(
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    q_len: int,
    k_len: int,
) -> torch.Tensor:
    if attention_mask.dim() != 4:
        raise ValueError(
            f"Expected attention_mask to be 4D, got {tuple(attention_mask.shape)}."
        )

    mask = attention_mask[:, :, :q_len, :k_len]
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(
            f"Unexpected batch dimension in attention_mask: {tuple(mask.shape)}."
        )
    if mask.shape[1] not in (1, num_heads):
        raise ValueError(
            f"Unexpected head dimension in attention_mask: {tuple(mask.shape)}."
        )
    if mask.shape[0] == 1 and batch_size != 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    if mask.shape[1] == 1 and num_heads != 1:
        mask = mask.expand(mask.shape[0], num_heads, -1, -1)
    return mask


def _resolve_causal_mask(
    attention_mask: torch.Tensor | None,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    k_len = key_states.shape[-2]
    q_len = query_states.shape[-2]
    if attention_mask is not None:
        return attention_mask[:, :, :, :k_len].to(target_dtype)

    offset = k_len - q_len
    mask = torch.full(
        (1, 1, q_len, k_len),
        torch.finfo(target_dtype).min,
        device=query_states.device,
        dtype=target_dtype,
    )
    q_pos = torch.arange(q_len, device=query_states.device).unsqueeze(1) + offset
    k_pos = torch.arange(k_len, device=query_states.device).unsqueeze(0)
    mask[..., q_pos >= k_pos] = 0
    return mask


def _mean_pool_attention_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")

    batch_size, num_heads, seq_len, head_dim = x.shape
    num_blocks = math.ceil(seq_len / block_size)
    pad = num_blocks * block_size - seq_len

    x_f = x.to(torch.float32)
    if pad > 0:
        x_f = F.pad(x_f, (0, 0, 0, pad))
    x_blocks = x_f.contiguous().view(
        batch_size, num_heads, num_blocks, block_size, head_dim
    )

    counts = torch.full(
        (num_blocks,),
        block_size,
        device=x.device,
        dtype=x_f.dtype,
    )
    if pad > 0:
        counts[-1] = block_size - pad
    return x_blocks.sum(dim=3) / counts.view(1, 1, num_blocks, 1)


def _block_validity_from_token_mask(
    allowed_token_mask: torch.Tensor,
    *,
    q_block: int,
    k_block: int,
) -> torch.Tensor:
    batch_size, num_heads, q_len, k_len = allowed_token_mask.shape
    q_blocks = math.ceil(q_len / q_block)
    k_blocks = math.ceil(k_len / k_block)
    q_pad = q_blocks * q_block - q_len
    k_pad = k_blocks * k_block - k_len

    allowed = allowed_token_mask.to(torch.uint8)
    if q_pad > 0 or k_pad > 0:
        allowed = F.pad(allowed, (0, k_pad, 0, q_pad), value=0)
    allowed = allowed.to(torch.bool)
    allowed_blocks = allowed.contiguous().view(
        batch_size,
        num_heads,
        q_blocks,
        q_block,
        k_blocks,
        k_block,
    )
    return allowed_blocks.any(dim=-1).any(dim=3)


def _select_topk_sparse_blocks(
    block_probs: torch.Tensor, valid_block_mask: torch.Tensor, *, topk: int
) -> torch.Tensor:
    width = block_probs.shape[-1]
    neg_large = torch.full_like(block_probs, -1e30)
    masked_probs = torch.where(valid_block_mask, block_probs, neg_large)
    sorted_probs, sorted_idx = masked_probs.sort(dim=-1, descending=True)
    del sorted_probs
    sorted_valid = valid_block_mask.gather(-1, sorted_idx)
    rank = torch.arange(width, device=block_probs.device).view(
        *([1] * (block_probs.dim() - 1)),
        width,
    )
    keep_sorted = sorted_valid & (rank < min(int(topk), width))
    keep_mask = torch.zeros_like(valid_block_mask)
    keep_mask.scatter_(-1, sorted_idx, keep_sorted)
    return keep_mask & valid_block_mask


def _expand_sparse_block_mask_to_token_mask(
    keep_block_mask: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    q_block: int,
    k_block: int,
) -> torch.Tensor:
    batch_size, num_heads, q_blocks, k_blocks = keep_block_mask.shape
    token_mask = (
        keep_block_mask[:, :, :, None, :, None]
        .expand(batch_size, num_heads, q_blocks, q_block, k_blocks, k_block)
        .reshape(batch_size, num_heads, q_blocks * q_block, k_blocks * k_block)
    )
    return token_mask[:, :, :q_len, :k_len]


def _build_sparse_attention_mask_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
) -> torch.Tensor:
    batch_size, num_heads, q_len, _ = query.shape
    k_len = key.shape[-2]
    mask_dtype = attention_mask.dtype

    base_mask = _broadcast_attention_mask(
        attention_mask,
        batch_size=batch_size,
        num_heads=num_heads,
        q_len=q_len,
        k_len=k_len,
    ).to(mask_dtype)
    mask_floor = torch.finfo(base_mask.dtype).min / 2
    allowed_token_mask = base_mask > mask_floor

    pooled_query = _mean_pool_attention_blocks(query, cfg.q_block)
    pooled_key = _mean_pool_attention_blocks(key, cfg.k_block)
    block_scores = torch.matmul(pooled_query, pooled_key.transpose(-1, -2)) * scaling

    valid_block_mask = _block_validity_from_token_mask(
        allowed_token_mask,
        q_block=cfg.q_block,
        k_block=cfg.k_block,
    )
    masked_scores = torch.where(
        valid_block_mask,
        block_scores,
        torch.full_like(block_scores, -1e30),
    )
    block_probs = torch.softmax(masked_scores, dim=-1)
    row_has_valid = valid_block_mask.any(dim=-1, keepdim=True)
    block_probs = torch.where(row_has_valid, block_probs, torch.zeros_like(block_probs))

    keep_block_mask = _select_topk_sparse_blocks(
        block_probs,
        valid_block_mask,
        topk=cfg.topk,
    )
    keep_token_mask = _expand_sparse_block_mask_to_token_mask(
        keep_block_mask,
        q_len=q_len,
        k_len=k_len,
        q_block=cfg.q_block,
        k_block=cfg.k_block,
    )
    keep_token_mask = keep_token_mask & allowed_token_mask
    return torch.where(
        keep_token_mask,
        base_mask,
        torch.full_like(base_mask, torch.finfo(base_mask.dtype).min),
    )


def _online_softmax_attention_with_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    pv_block_size: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if pv_block_size <= 0:
        raise ValueError(f"pv_block_size must be > 0, got {pv_block_size}.")

    batch_size, num_heads, q_len, head_dim = query.shape
    kv_len = key.shape[2]
    compute_dtype = torch.float32
    query_f = query.to(compute_dtype)
    key_f = key.to(compute_dtype)
    value_f = value.to(compute_dtype)

    running_max = torch.full(
        (batch_size, num_heads, q_len),
        float("-inf"),
        device=query.device,
        dtype=compute_dtype,
    )
    running_denom = torch.zeros(
        (batch_size, num_heads, q_len),
        device=query.device,
        dtype=compute_dtype,
    )
    running_out = torch.zeros(
        (batch_size, num_heads, q_len, head_dim),
        device=query.device,
        dtype=compute_dtype,
    )

    mask = attention_mask.to(compute_dtype) if attention_mask is not None else None
    for k_start in range(0, kv_len, pv_block_size):
        k_end = min(k_start + pv_block_size, kv_len)
        key_block = key_f[:, :, k_start:k_end, :]
        value_block = value_f[:, :, k_start:k_end, :]
        scores = torch.matmul(query_f, key_block.transpose(-1, -2)) * scaling
        if mask is not None:
            scores = scores + mask[..., k_start:k_end]

        block_max = scores.amax(dim=-1)
        valid_rows = torch.isfinite(block_max)
        safe_block_max = torch.where(valid_rows, block_max, torch.zeros_like(block_max))
        new_max = torch.maximum(running_max, block_max)

        prev_scale = torch.exp(running_max - new_max)
        block_scale = torch.exp(safe_block_max - new_max)
        block_probs_unnorm = torch.where(
            valid_rows.unsqueeze(-1),
            torch.exp(scores - safe_block_max.unsqueeze(-1)),
            torch.zeros_like(scores),
        )
        block_denom = block_probs_unnorm.sum(dim=-1) * block_scale
        running_out = (
            running_out * prev_scale.unsqueeze(-1)
            + torch.matmul(block_probs_unnorm, value_block)
            * block_scale.unsqueeze(-1)
        )
        running_denom = running_denom * prev_scale + block_denom
        running_max = new_max

    output = running_out / running_denom.clamp_min(1e-20).unsqueeze(-1)
    target_dtype = query.dtype if output_dtype is None else output_dtype
    return output.to(target_dtype)


def run_sparse_prefill_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the prototype PyTorch sparse attention path for one full-prefill request.

    Input shapes:
    - query: [q_len, num_heads, head_dim]
    - key/value: [k_len, num_kv_heads, head_dim]
    Returns:
    - output: [q_len, num_heads, head_dim]
    """

    query_states = query.transpose(0, 1).unsqueeze(0)
    key_states = key.transpose(0, 1).unsqueeze(0)
    value_states = value.transpose(0, 1).unsqueeze(0)

    if key_states.shape[1] != query_states.shape[1]:
        if query_states.shape[1] % key_states.shape[1] != 0:
            raise ValueError(
                "Sparse prefill attention requires query heads to be divisible "
                f"by KV heads, got num_heads={query_states.shape[1]} and "
                f"num_kv_heads={key_states.shape[1]}."
            )
        n_rep = query_states.shape[1] // key_states.shape[1]
        key_states = _repeat_kv(key_states, n_rep)
        value_states = _repeat_kv(value_states, n_rep)

    base_mask = _resolve_causal_mask(
        attention_mask=None,
        query_states=query_states,
        key_states=key_states,
        target_dtype=query_states.dtype,
    )
    sparse_mask = _build_sparse_attention_mask_topk(
        query_states,
        key_states,
        base_mask,
        scaling=scaling,
        cfg=cfg,
    )
    output = _online_softmax_attention_with_mask(
        query_states,
        key_states,
        value_states,
        sparse_mask,
        scaling=scaling,
        pv_block_size=cfg.pv_block_size,
        output_dtype=output_dtype,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for env-gated sparse prefill attention prototypes."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

AUTOPTQ_VLLM_SPARSE_ENABLE_ENV = "AUTOPTQ_VLLM_SPARSE_ENABLE"
AUTOPTQ_SPARSE_RUNTIME_JSON_ENV = "AUTOPTQ_SPARSE_RUNTIME_JSON"
AUTOPTQ_SPARSE_RUNTIME_KEY_ENV = "AUTOPTQ_SPARSE_RUNTIME_KEY"
AUTOPTQ_VLLM_SPARSE_IMPL_ENV = "AUTOPTQ_VLLM_SPARSE_IMPL"
AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV = (
    "AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE"
)

DEFAULT_PV_BLOCK_SIZE = 128
SparsePrefillRetainScoreLogMode = Literal["off", "summary", "layer", "head"]

logger = init_logger(__name__)


@dataclass(frozen=True)
class SparsePrefillTopKConfig:
    key: str
    name: str
    q_block: int
    k_block: int
    topk: int
    pv_block_size: int = DEFAULT_PV_BLOCK_SIZE


@dataclass(frozen=True)
class SparsePrefillSelectionStats:
    total_valid_blocks: int
    total_kept_blocks: int
    total_valid_rows: int
    retained_attention_score_sum: float
    per_head_total_valid_blocks: tuple[int, ...] = ()
    per_head_total_kept_blocks: tuple[int, ...] = ()
    per_head_total_valid_rows: tuple[int, ...] = ()
    per_head_retained_attention_score_sum: tuple[float, ...] = ()

    @property
    def density(self) -> float:
        if self.total_valid_blocks <= 0:
            return 0.0
        return float(self.total_kept_blocks / self.total_valid_blocks)

    @property
    def retained_attention_score_mean(self) -> float:
        if self.total_valid_rows <= 0:
            return 0.0
        return float(self.retained_attention_score_sum / self.total_valid_rows)

    @property
    def per_head_density(self) -> tuple[float, ...]:
        return tuple(
            0.0
            if valid_blocks <= 0
            else float(kept_blocks / valid_blocks)
            for kept_blocks, valid_blocks in zip(
                self.per_head_total_kept_blocks,
                self.per_head_total_valid_blocks,
            )
        )

    @property
    def per_head_retained_attention_score_mean(self) -> tuple[float, ...]:
        return tuple(
            0.0
            if valid_rows <= 0
            else float(retained_sum / valid_rows)
            for retained_sum, valid_rows in zip(
                self.per_head_retained_attention_score_sum,
                self.per_head_total_valid_rows,
            )
        )


def _collapse_per_head_counts(values: torch.Tensor) -> tuple[int, ...]:
    head_dim = 1 if values.dim() > 1 else 0
    reduce_dims = tuple(dim for dim in range(values.dim()) if dim != head_dim)
    per_head = values.to(torch.int64)
    if reduce_dims:
        per_head = per_head.sum(dim=reduce_dims)
    return tuple(int(v) for v in per_head.reshape(-1).tolist())


def _collapse_per_head_floats(values: torch.Tensor) -> tuple[float, ...]:
    head_dim = 1 if values.dim() > 1 else 0
    reduce_dims = tuple(dim for dim in range(values.dim()) if dim != head_dim)
    per_head = values.to(torch.float64)
    if reduce_dims:
        per_head = per_head.sum(dim=reduce_dims)
    return tuple(float(v) for v in per_head.reshape(-1).tolist())


def build_sparse_prefill_selection_stats(
    *,
    valid_block_counts: torch.Tensor,
    kept_block_counts: torch.Tensor,
    valid_row_mask: torch.Tensor,
    retained_attention_score_mass: torch.Tensor,
) -> SparsePrefillSelectionStats:
    total_valid_rows = int(valid_row_mask.sum().item())
    retained_sum = (
        float(retained_attention_score_mass.sum().item())
        if total_valid_rows > 0
        else 0.0
    )
    return SparsePrefillSelectionStats(
        total_valid_blocks=int(valid_block_counts.sum().item()),
        total_kept_blocks=int(kept_block_counts.sum().item()),
        total_valid_rows=total_valid_rows,
        retained_attention_score_sum=retained_sum,
        per_head_total_valid_blocks=_collapse_per_head_counts(valid_block_counts),
        per_head_total_kept_blocks=_collapse_per_head_counts(kept_block_counts),
        per_head_total_valid_rows=_collapse_per_head_counts(
            valid_row_mask.to(torch.int32)
        ),
        per_head_retained_attention_score_sum=_collapse_per_head_floats(
            retained_attention_score_mass
        ),
    )


def format_sparse_prefill_per_head_payload(
    selection_stats: SparsePrefillSelectionStats,
) -> dict[str, str]:
    return {
        "retain_scores": json.dumps(
            [
                round(float(value), 6)
                for value in selection_stats.per_head_retained_attention_score_mean
            ],
            separators=(",", ":"),
        ),
        "densities": json.dumps(
            [round(float(value), 6) for value in selection_stats.per_head_density],
            separators=(",", ":"),
        ),
        "valid_rows": json.dumps(
            list(selection_stats.per_head_total_valid_rows),
            separators=(",", ":"),
        ),
        "kept_blocks": json.dumps(
            list(selection_stats.per_head_total_kept_blocks),
            separators=(",", ":"),
        ),
        "valid_blocks": json.dumps(
            list(selection_stats.per_head_total_valid_blocks),
            separators=(",", ":"),
        ),
    }


def resolve_sparse_prefill_layer_info(
    layer: object | None,
) -> tuple[int | None, str | None]:
    layer_name = getattr(layer, "layer_name", None)
    if layer_name is None:
        return None, None

    layer_name = str(layer_name).strip()
    if not layer_name:
        return None, None

    try:
        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = extract_layer_index(layer_name)
    except (AssertionError, ImportError, ValueError):
        layer_idx = None
    return layer_idx, layer_name


def _parse_sparse_prefill_retain_score_log_mode(
    value: str | None,
) -> SparsePrefillRetainScoreLogMode:
    if value is None or not value.strip():
        return "off"

    normalized = value.strip().lower()
    mode_map: dict[str, SparsePrefillRetainScoreLogMode] = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "head",
        "true": "head",
        "yes": "head",
        "on": "head",
        "summary": "summary",
        "layer": "layer",
        "head": "head",
    }
    if normalized not in mode_map:
        raise ValueError(
            f"{AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV} must be one of "
            "'0', '1', 'summary', 'layer', or 'head'."
        )
    return mode_map[normalized]


def _parse_env_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_sparse_prefill_enabled() -> bool:
    return _parse_env_flag(os.getenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV))


def get_sparse_prefill_retain_score_log_mode() -> SparsePrefillRetainScoreLogMode:
    return _parse_sparse_prefill_retain_score_log_mode(
        os.getenv(AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV)
    )


def is_sparse_prefill_retain_score_recording_enabled() -> bool:
    return get_sparse_prefill_retain_score_log_mode() != "off"


def resolve_sparse_prefill_retain_score_log_mode(
    *,
    record_retain_score: bool,
    retain_score_log_mode: SparsePrefillRetainScoreLogMode | None,
) -> SparsePrefillRetainScoreLogMode:
    if retain_score_log_mode is not None:
        return retain_score_log_mode
    return "head" if record_retain_score else "off"


def should_log_sparse_prefill_layer_info(
    retain_score_log_mode: SparsePrefillRetainScoreLogMode,
) -> bool:
    return retain_score_log_mode in {"layer", "head"}


def should_log_sparse_prefill_per_head_stats(
    retain_score_log_mode: SparsePrefillRetainScoreLogMode,
) -> bool:
    return retain_score_log_mode == "head"


def get_sparse_prefill_impl_mode() -> str:
    value = os.getenv(AUTOPTQ_VLLM_SPARSE_IMPL_ENV)
    if value is None or not value.strip():
        return "auto"

    mode = value.strip().lower()
    if mode not in {"auto", "triton", "torch"}:
        raise ValueError(
            f"{AUTOPTQ_VLLM_SPARSE_IMPL_ENV} must be one of "
            "'auto', 'triton', or 'torch'."
        )
    return mode


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


def is_cached_prefix_prefill_request(query_len: int, seq_len: int) -> bool:
    if query_len <= 1:
        return False
    return query_len < seq_len


def is_sparse_prefill_request(query_len: int, seq_len: int) -> bool:
    return is_full_prefill_request(query_len, seq_len) or is_cached_prefix_prefill_request(
        query_len, seq_len
    )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    )


def _flatten_flash_kv_cache(
    kv_cache: torch.Tensor | None,
    *,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    if kv_cache is None or kv_cache.numel() == 0:
        return None, None, 0
    if kv_cache.dim() != 5 or kv_cache.shape[0] != 2:
        raise NotImplementedError(
            "Sparse prefill backend currently expects flash-style KV cache with "
            f"shape [2, num_blocks, block_size, num_kv_heads, head_dim], got {tuple(kv_cache.shape)}."
        )
    if str(kv_cache_dtype).startswith("fp8"):
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    block_size = int(kv_cache.shape[2])
    key_cache = kv_cache[0].reshape(-1, num_kv_heads, head_dim)
    value_cache = kv_cache[1].reshape(-1, num_kv_heads, head_dim)
    return key_cache, value_cache, block_size


def _reconstruct_sequence_slots(
    block_table_row: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")

    num_blocks = math.ceil(seq_len / block_size)
    if block_table_row.numel() < num_blocks:
        raise ValueError(
            "Sparse prefill backend requires enough block-table entries to "
            f"reconstruct the full sequence, got {block_table_row.numel()} entries "
            f"for seq_len={seq_len} and block_size={block_size}."
        )

    blocks = block_table_row.reshape(-1)[:num_blocks].to(dtype=torch.long)
    token_positions = torch.arange(seq_len, device=blocks.device, dtype=torch.long)
    block_offsets = torch.div(token_positions, block_size, rounding_mode="floor")
    token_offsets = token_positions.remainder(block_size)
    return blocks.index_select(0, block_offsets) * block_size + token_offsets


def gather_full_sequence_kv_from_paged_cache(
    *,
    kv_cache: torch.Tensor | None,
    block_table_row: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache, block_size = _flatten_flash_kv_cache(
        kv_cache,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_cache_dtype=kv_cache_dtype,
    )
    if key_cache is None or value_cache is None or block_size <= 0:
        raise ValueError(
            "Sparse prefill backend could not read the paged KV cache for a "
            "cached-prefix request."
        )
    if block_table_row.numel() == 0:
        raise ValueError(
            "Sparse prefill backend requires a non-empty block-table row for "
            "cached-prefix requests."
        )

    slots = _reconstruct_sequence_slots(
        block_table_row.to(device=key_cache.device),
        seq_len=seq_len,
        block_size=block_size,
    )
    full_key = key_cache.index_select(0, slots)
    full_value = value_cache.index_select(0, slots)
    return full_key, full_value


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


def _summarize_sparse_block_selection(
    block_probs: torch.Tensor,
    valid_block_mask: torch.Tensor,
    keep_block_mask: torch.Tensor,
) -> SparsePrefillSelectionStats:
    valid_block_counts = valid_block_mask.sum(dim=-1).to(torch.int32)
    kept_block_counts = keep_block_mask.sum(dim=-1).to(torch.int32)
    valid_row_mask = valid_block_counts > 0

    retained_attention_score_mass = torch.where(
        keep_block_mask,
        block_probs,
        torch.zeros_like(block_probs),
    ).sum(dim=-1)
    retained_attention_score_mass = torch.where(
        valid_row_mask,
        retained_attention_score_mass,
        torch.zeros_like(retained_attention_score_mass),
    )

    return build_sparse_prefill_selection_stats(
        valid_block_counts=valid_block_counts,
        kept_block_counts=kept_block_counts,
        valid_row_mask=valid_row_mask,
        retained_attention_score_mass=retained_attention_score_mass,
    )


def _build_sparse_attention_mask_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    return_selection_stats: bool = False,
) -> tuple[torch.Tensor, SparsePrefillSelectionStats | None]:
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
    selection_stats = (
        _summarize_sparse_block_selection(
            block_probs,
            valid_block_mask,
            keep_block_mask,
        )
        if return_selection_stats
        else None
    )
    return torch.where(
        keep_token_mask,
        base_mask,
        torch.full_like(base_mask, torch.finfo(base_mask.dtype).min),
    ), selection_stats


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
    record_retain_score: bool = False,
    retain_score_log_mode: SparsePrefillRetainScoreLogMode | None = None,
    layer: object | None = None,
) -> torch.Tensor:
    """Run the prototype PyTorch sparse attention path for one prefill request.

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
    resolved_log_mode = resolve_sparse_prefill_retain_score_log_mode(
        record_retain_score=record_retain_score,
        retain_score_log_mode=retain_score_log_mode,
    )
    sparse_mask, selection_stats = _build_sparse_attention_mask_topk(
        query_states,
        key_states,
        base_mask,
        scaling=scaling,
        cfg=cfg,
        return_selection_stats=resolved_log_mode != "off",
    )
    if selection_stats is not None:
        layer_idx = None
        layer_name = None
        if should_log_sparse_prefill_layer_info(resolved_log_mode):
            layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)
            logger.info(
                "Sparse prefill torch retain-score stats: layer_idx=%s "
                "layer_name=%s q_len=%d k_len=%d q_block=%d k_block=%d topk=%d "
                "avg_retain_score=%.6f density=%.6f valid_rows=%d "
                "kept_blocks=%d valid_blocks=%d.",
                layer_idx if layer_idx is not None else "NA",
                layer_name or "unknown",
                int(query.shape[0]),
                int(key.shape[0]),
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
                "Sparse prefill torch retain-score stats: q_len=%d k_len=%d "
                "q_block=%d k_block=%d topk=%d avg_retain_score=%.6f "
                "density=%.6f valid_rows=%d kept_blocks=%d valid_blocks=%d.",
                int(query.shape[0]),
                int(key.shape[0]),
                int(cfg.q_block),
                int(cfg.k_block),
                int(cfg.topk),
                selection_stats.retained_attention_score_mean,
                selection_stats.density,
                selection_stats.total_valid_rows,
                selection_stats.total_kept_blocks,
                selection_stats.total_valid_blocks,
            )
        if should_log_sparse_prefill_per_head_stats(resolved_log_mode):
            per_head_payload = format_sparse_prefill_per_head_payload(selection_stats)
            logger.info(
                "Sparse prefill torch retain-score per-head stats: layer_idx=%s "
                "layer_name=%s q_len=%d k_len=%d q_block=%d k_block=%d topk=%d "
                "num_heads=%d retain_scores=%s densities=%s valid_rows=%s "
                "kept_blocks=%s valid_blocks=%s.",
                layer_idx if layer_idx is not None else "NA",
                layer_name or "unknown",
                int(query.shape[0]),
                int(key.shape[0]),
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

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
SparsePrefillSelectionMode = Literal["topk", "threshold"]
SparsePrefillGQASharedMode = Literal["none", "mean", "mean_before"]

logger = init_logger(__name__)


@dataclass(frozen=True)
class SparsePrefillTopKConfig:
    key: str
    name: str
    q_block: int
    k_block: int
    topk: int | None = None
    threshold: float | None = None
    sink_block: int = 0
    sliding_window_block: int = 0
    gqa_shared: SparsePrefillGQASharedMode = "none"
    xattn: bool = False
    xattn_stride: int | None = None
    pv_block_size: int = DEFAULT_PV_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.q_block <= 0:
            raise ValueError(f"q_block must be > 0, got {self.q_block}.")
        if self.k_block <= 0:
            raise ValueError(f"k_block must be > 0, got {self.k_block}.")
        if self.sink_block < 0:
            raise ValueError(f"sink_block must be >= 0, got {self.sink_block}.")
        if self.sliding_window_block < 0:
            raise ValueError(
                "sliding_window_block must be >= 0, "
                f"got {self.sliding_window_block}."
            )

        has_topk = self.topk is not None
        has_threshold = self.threshold is not None
        if has_topk == has_threshold:
            raise ValueError(
                "Sparse prefill config must define exactly one of 'topk' "
                "or 'threshold'."
            )
        if has_topk and (
            isinstance(self.topk, bool) or int(self.topk) <= 0  # type: ignore[arg-type]
        ):
            raise ValueError(f"topk must be > 0, got {self.topk}.")
        if has_threshold and (
            isinstance(self.threshold, bool)
            or float(self.threshold) <= 0.0  # type: ignore[arg-type]
            or float(self.threshold) > 1.0  # type: ignore[arg-type]
        ):
            raise ValueError(
                "threshold must be in the range (0, 1], "
                f"got {self.threshold}."
            )
        if self.gqa_shared not in {"none", "mean", "mean_before"}:
            raise ValueError(
                "Sparse scheme field 'gqa_shared' must be one of "
                "'none', 'mean', or 'mean_before'."
            )
        if self.xattn:
            if self.gqa_shared == "mean_before":
                raise ValueError(
                    "Sparse scheme field 'gqa_shared'='mean_before' currently does "
                    "not support 'xattn'=true."
                )
            if self.xattn_stride is None:
                raise ValueError(
                    "Sparse xattn configs require a positive 'xattn_stride'."
                )
            if self.xattn_stride <= 0:
                raise ValueError(
                    f"xattn_stride must be > 0, got {self.xattn_stride}."
                )
            if self.q_block % self.xattn_stride != 0:
                raise ValueError(
                    f"q_block={self.q_block} must be divisible by "
                    f"xattn_stride={self.xattn_stride}."
                )
            if self.k_block % self.xattn_stride != 0:
                raise ValueError(
                    f"k_block={self.k_block} must be divisible by "
                    f"xattn_stride={self.xattn_stride}."
                )
        elif self.xattn_stride is not None:
            raise ValueError(
                "Sparse scheme field 'xattn_stride' requires 'xattn'=true."
            )

    @property
    def selection_mode(self) -> SparsePrefillSelectionMode:
        return "threshold" if self.threshold is not None else "topk"

    @property
    def max_selected_blocks(self) -> int:
        base = int(self.topk) if self.topk is not None else 0
        return base + int(self.sink_block) + int(self.sliding_window_block)


def format_sparse_prefill_selection_policy(cfg: SparsePrefillTopKConfig) -> str:
    if cfg.threshold is not None:
        policy = f"threshold={cfg.threshold:g}"
    else:
        policy = f"topk={cfg.topk}"
    if cfg.gqa_shared != "none":
        policy = f"{policy},gqa_shared={cfg.gqa_shared}"
    if cfg.xattn:
        policy = f"{policy},xattn_stride={cfg.xattn_stride}"
    return policy


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


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Sparse scheme field '{field_name}' must be an integer.")
    if value < 0:
        raise ValueError(f"Sparse scheme field '{field_name}' must be >= 0.")
    return int(value)


def _require_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Sparse scheme field '{field_name}' must be a float in the range "
            "(0, 1]."
        )
    value_f = float(value)
    if value_f <= 0.0 or value_f > 1.0:
        raise ValueError(
            f"Sparse scheme field '{field_name}' must be in the range (0, 1]."
        )
    return value_f


def _parse_sparse_prefill_xattn_flag(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("Sparse scheme field 'xattn' must be a boolean.")
    return value


def _parse_sparse_prefill_gqa_shared(value: object) -> SparsePrefillGQASharedMode:
    if value is None:
        return "none"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Sparse scheme field 'gqa_shared' must be a non-empty string."
        )
    normalized = value.strip().lower()
    if normalized not in {"none", "mean", "mean_before"}:
        raise ValueError(
            "Sparse scheme field 'gqa_shared' must be one of "
            "'none', 'mean', or 'mean_before'."
        )
    return normalized  # type: ignore[return-value]


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

    if scheme.get("ratio") is not None:
        raise ValueError(
            f"Only topk and threshold sparse schemes are supported in vLLM v1. "
            f"Scheme '{key}' must not define 'ratio'."
        )
    topk_value = scheme.get("topk")
    threshold_value = scheme.get("threshold")
    if (topk_value is None) == (threshold_value is None):
        raise ValueError(
            f"Sparse scheme '{key}' must define exactly one of 'topk' "
            f"or 'threshold'."
        )

    q_block = _require_positive_int(scheme.get("q_block"), "q_block")
    k_block = _require_positive_int(scheme.get("k_block"), "k_block")
    topk = (
        _require_positive_int(topk_value, "topk") if topk_value is not None else None
    )
    threshold = (
        _require_probability(threshold_value, "threshold")
        if threshold_value is not None
        else None
    )
    if scheme.get("q_pooling") is not None:
        raise ValueError(
            "Sparse scheme field 'q_pooling' is no longer supported."
        )
    gqa_shared = _parse_sparse_prefill_gqa_shared(scheme.get("gqa_shared"))
    xattn = _parse_sparse_prefill_xattn_flag(scheme.get("xattn"))
    xattn_stride_value = scheme.get("xattn_stride")
    xattn_stride = (
        _require_positive_int(xattn_stride_value, "xattn_stride")
        if xattn_stride_value is not None
        else None
    )

    return SparsePrefillTopKConfig(
        key=key,
        name=str(scheme.get("name", key)),
        q_block=q_block,
        k_block=k_block,
        topk=topk,
        threshold=threshold,
        sink_block=_require_nonnegative_int(
            scheme.get("sink_block", 0),
            "sink_block",
        ),
        sliding_window_block=_require_nonnegative_int(
            scheme.get("sliding_window_block", 0),
            "sliding_window_block",
        ),
        gqa_shared=gqa_shared,
        xattn=xattn,
        xattn_stride=xattn_stride,
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


def _reshape_xattn_sequence(
    x: torch.Tensor,
    *,
    stride: int,
    reverse: bool,
) -> tuple[torch.Tensor, int]:
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}.")

    batch_size, num_heads, seq_len, head_dim = x.shape
    reduced_len = math.ceil(seq_len / stride)
    pad = reduced_len * stride - seq_len
    x_f = x.to(torch.float32)
    if pad > 0:
        x_f = F.pad(x_f, (0, 0, 0, pad))

    offsets = range(stride - 1, -1, -1) if reverse else range(stride)
    pieces = [x_f[:, :, offset::stride, :] for offset in offsets]
    reshaped = torch.cat(pieces, dim=-1)
    expected_shape = (batch_size, num_heads, reduced_len, head_dim * stride)
    if reshaped.shape != expected_shape:
        raise RuntimeError(
            "Unexpected xattn reshaped state shape, expected "
            f"{expected_shape} but got {tuple(reshaped.shape)}."
        )
    return reshaped, reduced_len


def _expand_head_mask(mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    if mask.shape[1] == num_heads:
        return mask
    if mask.shape[1] != 1:
        raise ValueError(
            f"Expected mask head dimension to be 1 or {num_heads}, got {mask.shape[1]}."
        )
    return mask.expand(mask.shape[0], num_heads, *mask.shape[2:])


def _apply_gqa_shared_pooled_query(
    pooled_query: torch.Tensor,
    *,
    num_kv_heads: int,
    cfg: SparsePrefillTopKConfig,
) -> tuple[torch.Tensor, bool]:
    if cfg.gqa_shared != "mean_before":
        return pooled_query, False
    if pooled_query.dim() != 4:
        raise ValueError(
            "gqa_shared pooled-query reduction expects a 4D tensor, got "
            f"{tuple(pooled_query.shape)}."
        )
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}.")

    batch_size, num_heads, q_blocks, head_dim = pooled_query.shape
    if num_heads == num_kv_heads:
        return pooled_query, False
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "gqa_shared pooled-query reduction requires query heads to be divisible "
            f"by KV heads, got num_heads={num_heads} and num_kv_heads={num_kv_heads}."
        )

    heads_per_group = num_heads // num_kv_heads
    grouped_query = pooled_query.reshape(
        batch_size, num_kv_heads, heads_per_group, q_blocks, head_dim
    )
    shared_query = grouped_query.mean(dim=2)
    expanded_query = (
        shared_query.unsqueeze(2)
        .expand(-1, -1, heads_per_group, -1, -1)
        .reshape_as(pooled_query)
    )
    return expanded_query, True


def _apply_gqa_shared_block_scores(
    block_scores: torch.Tensor,
    valid_block_mask: torch.Tensor,
    *,
    num_kv_heads: int,
    cfg: SparsePrefillTopKConfig,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if cfg.gqa_shared != "mean":
        return block_scores, valid_block_mask, False
    if block_scores.dim() != 4 or valid_block_mask.dim() != 4:
        raise ValueError(
            "gqa_shared block-score reduction expects 4D tensors, got "
            f"{tuple(block_scores.shape)} and {tuple(valid_block_mask.shape)}."
        )
    if block_scores.shape != valid_block_mask.shape:
        raise ValueError(
            "gqa_shared block-score reduction requires score/mask shapes to match, got "
            f"{tuple(block_scores.shape)} and {tuple(valid_block_mask.shape)}."
        )
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}.")

    batch_size, num_heads, q_blocks, k_blocks = block_scores.shape
    if num_heads == num_kv_heads:
        return block_scores, valid_block_mask, False
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "gqa_shared sparse selection requires query heads to be divisible by "
            f"KV heads, got num_heads={num_heads} and num_kv_heads={num_kv_heads}."
        )

    heads_per_group = num_heads // num_kv_heads
    grouped_scores = block_scores.reshape(
        batch_size, num_kv_heads, heads_per_group, q_blocks, k_blocks
    )
    grouped_valid = valid_block_mask.reshape(
        batch_size, num_kv_heads, heads_per_group, q_blocks, k_blocks
    )

    valid_counts = grouped_valid.to(grouped_scores.dtype).sum(dim=2).clamp_min(1.0)
    shared_scores = torch.where(
        grouped_valid, grouped_scores, torch.zeros_like(grouped_scores)
    ).sum(dim=2) / valid_counts
    shared_valid = grouped_valid.any(dim=2)

    expanded_scores = (
        shared_scores.unsqueeze(2)
        .expand(-1, -1, heads_per_group, -1, -1)
        .reshape_as(block_scores)
    )
    expanded_valid = (
        shared_valid.unsqueeze(2)
        .expand(-1, -1, heads_per_group, -1, -1)
        .reshape_as(valid_block_mask)
    )
    return expanded_scores, expanded_valid, True


def _build_xattn_block_scores_and_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    reduced_valid_mask: torch.Tensor,
    final_valid_block_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not cfg.xattn or cfg.xattn_stride is None:
        raise ValueError("xattn score construction requires cfg.xattn_stride.")
    if query.shape[1] != key.shape[1]:
        raise ValueError(
            "xattn score construction requires query and key to have the same "
            f"number of heads, got {query.shape[1]} and {key.shape[1]}."
        )

    stride = int(cfg.xattn_stride)
    batch_size, num_heads, q_len, _ = query.shape
    k_len = key.shape[-2]
    q_group = int(cfg.q_block) // stride
    k_group = int(cfg.k_block) // stride
    q_blocks = math.ceil(q_len / int(cfg.q_block))
    k_blocks = math.ceil(k_len / int(cfg.k_block))

    reshaped_query, q_reduced_len = _reshape_xattn_sequence(
        query, stride=stride, reverse=True
    )
    reshaped_key, k_reduced_len = _reshape_xattn_sequence(
        key, stride=stride, reverse=False
    )
    reduced_scores = torch.matmul(
        reshaped_query, reshaped_key.transpose(-1, -2)
    ) * (float(scaling) / float(stride))

    reduced_valid_mask = _expand_head_mask(
        reduced_valid_mask.to(device=query.device, dtype=torch.bool),
        num_heads,
    )
    final_valid_block_mask = _expand_head_mask(
        final_valid_block_mask.to(device=query.device, dtype=torch.bool),
        num_heads,
    )
    if reduced_valid_mask.shape[-2:] != (q_reduced_len, k_reduced_len):
        raise ValueError(
            "Unexpected reduced_valid_mask shape for xattn, got "
            f"{tuple(reduced_valid_mask.shape)} expected "
            f"(*, {q_reduced_len}, {k_reduced_len})."
        )
    if final_valid_block_mask.shape[-2:] != (q_blocks, k_blocks):
        raise ValueError(
            "Unexpected final_valid_block_mask shape for xattn, got "
            f"{tuple(final_valid_block_mask.shape)} expected "
            f"(*, {q_blocks}, {k_blocks})."
        )

    neg_large = torch.full_like(reduced_scores, -1e30)
    masked_reduced_scores = torch.where(reduced_valid_mask, reduced_scores, neg_large)
    reduced_row_has_valid = reduced_valid_mask.any(dim=-1, keepdim=True)
    reduced_probs = torch.where(
        reduced_row_has_valid,
        torch.softmax(masked_reduced_scores, dim=-1),
        torch.zeros_like(masked_reduced_scores),
    )

    q_pad = q_blocks * q_group - q_reduced_len
    k_pad = k_blocks * k_group - k_reduced_len
    if q_pad < 0 or k_pad < 0:
        raise ValueError(
            "xattn reduced grid exceeded configured block layout with "
            f"q_pad={q_pad}, k_pad={k_pad}."
        )

    score_values = torch.where(
        reduced_valid_mask,
        reduced_scores,
        torch.zeros_like(reduced_scores),
    )
    valid_pad = reduced_valid_mask.to(torch.uint8)
    query_row_valid = torch.ones(
        (batch_size, num_heads, q_reduced_len),
        device=query.device,
        dtype=torch.uint8,
    )
    if q_pad > 0 or k_pad > 0:
        score_values = F.pad(score_values, (0, k_pad, 0, q_pad), value=0.0)
        reduced_probs = F.pad(reduced_probs, (0, k_pad, 0, q_pad), value=0.0)
        valid_pad = F.pad(valid_pad, (0, k_pad, 0, q_pad), value=0)
        query_row_valid = F.pad(query_row_valid, (0, q_pad), value=0)

    score_blocks = score_values.contiguous().view(
        batch_size, num_heads, q_blocks, q_group, k_blocks, k_group
    )
    prob_blocks = reduced_probs.contiguous().view(
        batch_size, num_heads, q_blocks, q_group, k_blocks, k_group
    )
    valid_blocks = valid_pad.to(torch.bool).contiguous().view(
        batch_size, num_heads, q_blocks, q_group, k_blocks, k_group
    )
    q_row_valid_blocks = query_row_valid.to(torch.bool).contiguous().view(
        batch_size, num_heads, q_blocks, q_group
    )

    block_score_counts = valid_blocks.sum(dim=5).sum(dim=3)
    block_scores = torch.where(
        valid_blocks,
        score_blocks,
        torch.zeros_like(score_blocks),
    ).sum(dim=5).sum(dim=3)
    block_scores = block_scores / block_score_counts.to(torch.float32).clamp_min(1.0)

    q_row_counts = q_row_valid_blocks.sum(dim=3).to(torch.float32).clamp_min(1.0)
    block_probs = torch.where(
        valid_blocks,
        prob_blocks,
        torch.zeros_like(prob_blocks),
    ).sum(dim=5).sum(dim=3)
    block_probs = block_probs / q_row_counts.unsqueeze(-1)

    block_scores = torch.where(
        final_valid_block_mask,
        block_scores,
        torch.zeros_like(block_scores),
    )
    block_probs = torch.where(
        final_valid_block_mask,
        block_probs,
        torch.zeros_like(block_probs),
    )
    return block_scores, block_probs, final_valid_block_mask


def _build_mandatory_sparse_block_mask(
    valid_block_mask: torch.Tensor,
    *,
    sink_block: int,
    sliding_window_block: int,
) -> torch.Tensor:
    if sink_block <= 0 and sliding_window_block <= 0:
        return torch.zeros_like(valid_block_mask)

    valid_int = valid_block_mask.to(torch.int32)
    valid_rank = torch.cumsum(valid_int, dim=-1) - 1
    keep_mask = torch.zeros_like(valid_block_mask)

    if sink_block > 0:
        keep_mask = keep_mask | (valid_block_mask & (valid_rank < int(sink_block)))

    if sliding_window_block > 0:
        valid_count = valid_int.sum(dim=-1, keepdim=True)
        window_start_rank = (valid_count - int(sliding_window_block)).clamp_min(0)
        keep_mask = keep_mask | (valid_block_mask & (valid_rank >= window_start_rank))

    return keep_mask


def _select_topk_sparse_blocks(
    block_values: torch.Tensor, valid_block_mask: torch.Tensor, *, topk: int
) -> torch.Tensor:
    width = block_values.shape[-1]
    k_keep = min(int(topk), width)
    if k_keep <= 0:
        return torch.zeros_like(valid_block_mask)

    neg_large = torch.full_like(block_values, float("-inf"))
    masked_values = torch.where(valid_block_mask, block_values, neg_large)
    topk_idx = torch.topk(masked_values, k=k_keep, dim=-1).indices
    counts = valid_block_mask.sum(dim=-1).clamp_max(k_keep).to(torch.int32)
    rank = torch.arange(k_keep, device=block_values.device, dtype=torch.int32).view(
        *([1] * (block_values.dim() - 1)),
        k_keep,
    )
    keep_selected = rank < counts.unsqueeze(-1)
    return _scatter_sparse_block_mask(
        topk_idx,
        keep_selected,
        width=width,
    ) & valid_block_mask


def _scatter_sparse_block_mask(
    block_idx: torch.Tensor,
    keep_selected: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    safe_block_idx = torch.where(keep_selected, block_idx, 0)
    # Padded ranks are mapped to index 0; use additive scatter so they cannot
    # overwrite a real selection at block 0 when counts < width.
    keep_mask = torch.zeros(
        (*block_idx.shape[:-1], width),
        device=block_idx.device,
        dtype=torch.int32,
    )
    keep_mask.scatter_add_(-1, safe_block_idx, keep_selected.to(torch.int32))
    return keep_mask > 0


def _select_threshold_sparse_blocks(
    block_probs: torch.Tensor,
    valid_block_mask: torch.Tensor,
    *,
    required_mass: torch.Tensor,
) -> torch.Tensor:
    if block_probs.shape != valid_block_mask.shape:
        raise ValueError(
            "block_probs and valid_block_mask must have the same shape, got "
            f"{tuple(block_probs.shape)} vs {tuple(valid_block_mask.shape)}."
        )

    width = block_probs.shape[-1]
    if width <= 0:
        return torch.zeros_like(valid_block_mask)

    neg_large = torch.full_like(block_probs, float("-inf"))
    masked_probs = torch.where(valid_block_mask, block_probs, neg_large)
    sorted_probs, sorted_idx = torch.sort(masked_probs, dim=-1, descending=True)
    candidate_counts = valid_block_mask.sum(dim=-1).to(torch.int32)
    rank = torch.arange(width, device=block_probs.device, dtype=torch.int32).view(
        *([1] * (block_probs.dim() - 1)),
        width,
    )
    keep_candidate = rank < candidate_counts.unsqueeze(-1)
    sorted_probs = torch.where(keep_candidate, sorted_probs, torch.zeros_like(sorted_probs))
    cumulative_without_self = torch.cat(
        [torch.zeros_like(sorted_probs[..., :1]), sorted_probs[..., :-1]],
        dim=-1,
    ).cumsum(dim=-1)
    keep_selected = keep_candidate & (
        cumulative_without_self < required_mass.to(block_probs.dtype)
    )
    return _scatter_sparse_block_mask(
        sorted_idx,
        keep_selected,
        width=width,
    ) & valid_block_mask


def _merge_mandatory_and_sparse_blocks(
    block_values: torch.Tensor,
    valid_block_mask: torch.Tensor,
    *,
    cfg: SparsePrefillTopKConfig,
    block_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mandatory_keep_mask = _build_mandatory_sparse_block_mask(
        valid_block_mask,
        sink_block=int(cfg.sink_block),
        sliding_window_block=int(cfg.sliding_window_block),
    )
    selectable_block_mask = valid_block_mask & ~mandatory_keep_mask
    if cfg.topk is not None:
        additional_keep_mask = _select_topk_sparse_blocks(
            block_values,
            selectable_block_mask,
            topk=int(cfg.topk),
        )
    else:
        if block_probs is None:
            raise ValueError("Threshold sparse selection requires block_probs.")
        mandatory_mass = torch.where(
            mandatory_keep_mask,
            block_probs,
            torch.zeros_like(block_probs),
        ).sum(dim=-1, keepdim=True)
        remaining_mass = (
            torch.full_like(mandatory_mass, float(cfg.threshold)) - mandatory_mass
        ).clamp_min(0.0)
        additional_keep_mask = _select_threshold_sparse_blocks(
            block_probs,
            selectable_block_mask,
            required_mass=remaining_mass,
        )
    keep_block_mask = mandatory_keep_mask | additional_keep_mask
    return keep_block_mask, keep_block_mask.sum(dim=-1).to(torch.int32)


def _keep_block_mask_to_block_indices(
    keep_block_mask: torch.Tensor,
    *,
    full_topk: int,
    counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if full_topk < 0:
        raise ValueError(f"full_topk must be >= 0, got {full_topk}.")
    if counts is None:
        counts = keep_block_mask.sum(dim=-1).to(torch.int32)

    if full_topk == 0:
        return torch.empty(
            (*keep_block_mask.shape[:-1], 0),
            device=keep_block_mask.device,
            dtype=torch.int32,
        )

    width = keep_block_mask.shape[-1]
    block_idx = torch.arange(width, device=keep_block_mask.device, dtype=torch.int32).view(
        *([1] * (keep_block_mask.dim() - 1)),
        width,
    )
    block_idx = block_idx.expand_as(keep_block_mask)
    sentinel = torch.full_like(block_idx, width)
    sorted_idx = torch.where(keep_block_mask, block_idx, sentinel).sort(dim=-1).values
    topk_idx = sorted_idx[..., :full_topk]
    clamped_counts = counts.clamp_max(full_topk).to(torch.int32)
    rank = torch.arange(full_topk, device=keep_block_mask.device, dtype=torch.int32).view(
        *([1] * (keep_block_mask.dim() - 1)),
        full_topk,
    )
    invalid = rank >= clamped_counts.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(topk_idx, -1), topk_idx)


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
    num_kv_heads: int | None = None,
    return_selection_stats: bool = False,
) -> tuple[torch.Tensor, SparsePrefillSelectionStats | None]:
    batch_size, num_heads, q_len, _ = query.shape
    k_len = key.shape[-2]
    mask_dtype = attention_mask.dtype
    resolved_num_kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)

    base_mask = _broadcast_attention_mask(
        attention_mask,
        batch_size=batch_size,
        num_heads=num_heads,
        q_len=q_len,
        k_len=k_len,
    ).to(mask_dtype)
    mask_floor = torch.finfo(base_mask.dtype).min / 2
    allowed_token_mask = base_mask > mask_floor

    block_probs = None
    if cfg.xattn:
        reduced_valid_mask = _block_validity_from_token_mask(
            allowed_token_mask,
            q_block=int(cfg.xattn_stride),
            k_block=int(cfg.xattn_stride),
        )
        final_valid_block_mask = _block_validity_from_token_mask(
            allowed_token_mask,
            q_block=cfg.q_block,
            k_block=cfg.k_block,
        )
        block_scores, block_probs, valid_block_mask = _build_xattn_block_scores_and_probs(
            query,
            key,
            scaling=float(scaling),
            cfg=cfg,
            reduced_valid_mask=reduced_valid_mask,
            final_valid_block_mask=final_valid_block_mask,
        )
    else:
        pooled_key = _mean_pool_attention_blocks(key, cfg.k_block)
        pooled_query = _mean_pool_attention_blocks(query, cfg.q_block)
        pooled_query, _ = _apply_gqa_shared_pooled_query(
            pooled_query,
            num_kv_heads=resolved_num_kv_heads,
            cfg=cfg,
        )
        block_scores = (
            torch.matmul(pooled_query, pooled_key.transpose(-1, -2)) * scaling
        )
        valid_block_mask = _block_validity_from_token_mask(
            allowed_token_mask,
            q_block=cfg.q_block,
            k_block=cfg.k_block,
        )

    block_scores, valid_block_mask, gqa_shared_applied = _apply_gqa_shared_block_scores(
        block_scores,
        valid_block_mask,
        num_kv_heads=resolved_num_kv_heads,
        cfg=cfg,
    )
    if gqa_shared_applied:
        block_probs = None

    masked_scores = torch.where(
        valid_block_mask,
        block_scores,
        torch.full_like(block_scores, -1e30),
    )
    if block_probs is None:
        block_probs = torch.softmax(masked_scores, dim=-1)
        row_has_valid = valid_block_mask.any(dim=-1, keepdim=True)
        block_probs = torch.where(
            row_has_valid, block_probs, torch.zeros_like(block_probs)
        )

    keep_block_mask, _ = _merge_mandatory_and_sparse_blocks(
        masked_scores,
        valid_block_mask,
        cfg=cfg,
        block_probs=block_probs,
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

    original_num_kv_heads = int(key.shape[1])
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
        num_kv_heads=original_num_kv_heads,
        return_selection_stats=resolved_log_mode != "off",
    )
    if selection_stats is not None:
        selection_policy = format_sparse_prefill_selection_policy(cfg)
        layer_idx = None
        layer_name = None
        if should_log_sparse_prefill_layer_info(resolved_log_mode):
            layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)
            logger.info(
                "Sparse prefill torch retain-score stats: layer_idx=%s "
                "layer_name=%s q_len=%d k_len=%d q_block=%d k_block=%d %s "
                "avg_retain_score=%.6f density=%.6f valid_rows=%d "
                "kept_blocks=%d valid_blocks=%d.",
                layer_idx if layer_idx is not None else "NA",
                layer_name or "unknown",
                int(query.shape[0]),
                int(key.shape[0]),
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
                "Sparse prefill torch retain-score stats: q_len=%d k_len=%d "
                "q_block=%d k_block=%d %s avg_retain_score=%.6f "
                "density=%.6f valid_rows=%d kept_blocks=%d valid_blocks=%d.",
                int(query.shape[0]),
                int(key.shape[0]),
                int(cfg.q_block),
                int(cfg.k_block),
                selection_policy,
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
                "layer_name=%s q_len=%d k_len=%d q_block=%d k_block=%d %s "
                "num_heads=%d retain_scores=%s densities=%s valid_rows=%s "
                "kept_blocks=%s valid_blocks=%s.",
                layer_idx if layer_idx is not None else "NA",
                layer_name or "unknown",
                int(query.shape[0]),
                int(key.shape[0]),
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

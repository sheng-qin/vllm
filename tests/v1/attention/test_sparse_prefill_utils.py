# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.v1.attention.backends.sparse_prefill_flash_attn import (
    _ensure_sparse_output_has_no_nan,
)
from vllm.v1.attention.backends.sparse_prefill_utils import (
    AUTOPTQ_SPARSE_RUNTIME_JSON_ENV,
    AUTOPTQ_SPARSE_RUNTIME_KEY_ENV,
    AUTOPTQ_VLLM_SPARSE_ENABLE_ENV,
    AUTOPTQ_VLLM_SPARSE_IMPL_ENV,
    _reconstruct_sequence_slots,
    gather_full_sequence_kv_from_paged_cache,
    get_sparse_prefill_impl_mode,
    get_sparse_prefill_topk_config,
    is_cached_prefix_prefill_request,
    is_full_prefill_request,
    is_sparse_prefill_request,
)


def test_get_sparse_prefill_topk_config_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, raising=False)
    monkeypatch.delenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, raising=False)
    monkeypatch.delenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, raising=False)

    assert get_sparse_prefill_topk_config() is None


def test_get_sparse_prefill_impl_mode_defaults_to_auto(monkeypatch):
    monkeypatch.delenv(AUTOPTQ_VLLM_SPARSE_IMPL_ENV, raising=False)

    assert get_sparse_prefill_impl_mode() == "auto"


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("auto", "auto"),
        ("triton", "triton"),
        ("torch", "torch"),
        ("  TRITON  ", "triton"),
    ],
)
def test_get_sparse_prefill_impl_mode_parses_valid_values(
    monkeypatch, env_value: str, expected: str
):
    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_IMPL_ENV, env_value)

    assert get_sparse_prefill_impl_mode() == expected


def test_get_sparse_prefill_impl_mode_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_IMPL_ENV, "cuda")

    with pytest.raises(ValueError, match="must be one of"):
        get_sparse_prefill_impl_mode()


def test_get_sparse_prefill_topk_config_parses_topk_scheme(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_v3": {
                    "name": "q_16_k_16_topk_128",
                    "q_block": 16,
                    "k_block": 16,
                    "topk": 128,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_v3")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.key == "sparse_v3"
    assert cfg.name == "q_16_k_16_topk_128"
    assert cfg.q_block == 16
    assert cfg.k_block == 16
    assert cfg.topk == 128


def test_sparse_output_nan_check_accepts_finite_output():
    output = torch.zeros((2, 3, 4), dtype=torch.float32)

    _ensure_sparse_output_has_no_nan(
        output,
        req_idx=0,
        query_len=2,
        seq_len=2,
        request_kind="full_prefill",
        impl_mode="triton",
    )


def test_sparse_output_nan_check_rejects_nan_output():
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    output[1, 2, 3] = float("nan")

    with pytest.raises(RuntimeError, match="produced NaN values in output"):
        _ensure_sparse_output_has_no_nan(
            output,
            req_idx=7,
            query_len=4,
            seq_len=16,
            request_kind="cached_prefix",
            impl_mode="torch",
        )


def test_get_sparse_prefill_topk_config_rejects_non_topk_scheme(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_v1": {
                    "name": "q_16_k_16_threshold_0.95",
                    "q_block": 16,
                    "k_block": 16,
                    "threshold": 0.95,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_v1")

    with pytest.raises(ValueError, match="Only topk sparse schemes are supported"):
        get_sparse_prefill_topk_config()


def test_reconstruct_sequence_slots_uses_block_table_order():
    block_table_row = torch.tensor([5, 7], dtype=torch.int32)

    slots = _reconstruct_sequence_slots(
        block_table_row,
        seq_len=6,
        block_size=4,
    )

    assert slots.tolist() == [20, 21, 22, 23, 28, 29]


def test_gather_full_sequence_kv_from_paged_cache_reconstructs_dense_sequence():
    num_blocks = 4
    block_size = 4
    num_kv_heads = 2
    head_dim = 3
    kv_cache = torch.empty(
        (2, num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.float32,
    )

    for block_idx in range(num_blocks):
        for token_offset in range(block_size):
            token_id = block_idx * block_size + token_offset
            kv_cache[0, block_idx, token_offset].fill_(float(token_id))
            kv_cache[1, block_idx, token_offset].fill_(float(token_id + 1000))

    full_key, full_value = gather_full_sequence_kv_from_paged_cache(
        kv_cache=kv_cache,
        block_table_row=torch.tensor([3, 1], dtype=torch.int32),
        seq_len=6,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_cache_dtype="auto",
    )

    assert tuple(full_key.shape) == (6, num_kv_heads, head_dim)
    assert tuple(full_value.shape) == (6, num_kv_heads, head_dim)
    assert full_key[:, 0, 0].tolist() == [12.0, 13.0, 14.0, 15.0, 4.0, 5.0]
    assert full_value[:, 0, 0].tolist() == [
        1012.0,
        1013.0,
        1014.0,
        1015.0,
        1004.0,
        1005.0,
    ]


def test_sparse_prefill_request_classifies_cached_prefix_and_decode():
    assert is_full_prefill_request(8, 8)
    assert not is_cached_prefix_prefill_request(8, 8)
    assert is_sparse_prefill_request(8, 8)

    assert is_cached_prefix_prefill_request(4, 8)
    assert is_sparse_prefill_request(4, 8)

    assert not is_sparse_prefill_request(1, 8)
    assert not is_cached_prefix_prefill_request(1, 8)
    assert not is_sparse_prefill_request(9, 8)

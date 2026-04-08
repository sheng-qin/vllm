# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1.attention.backends.sparse_prefill_utils import (
    AUTOPTQ_SPARSE_RUNTIME_JSON_ENV,
    AUTOPTQ_SPARSE_RUNTIME_KEY_ENV,
    AUTOPTQ_VLLM_SPARSE_ENABLE_ENV,
    get_sparse_prefill_topk_config,
)


def test_get_sparse_prefill_topk_config_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, raising=False)
    monkeypatch.delenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, raising=False)
    monkeypatch.delenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, raising=False)

    assert get_sparse_prefill_topk_config() is None


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

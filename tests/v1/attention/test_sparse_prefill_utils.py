# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import sparse_prefill_utils as sparse_prefill_utils_module
from vllm.v1.attention.backends.sparse_prefill_flash_attn import (
    AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV,
    _parse_optional_env_flag,
    _ensure_sparse_output_has_no_nan,
)
from vllm.v1.attention.backends.sparse_prefill_utils import (
    AUTOPTQ_SPARSE_RUNTIME_JSON_ENV,
    AUTOPTQ_SPARSE_RUNTIME_KEY_ENV,
    AUTOPTQ_VLLM_SPARSE_ENABLE_ENV,
    AUTOPTQ_VLLM_SPARSE_IMPL_ENV,
    AUTOPTQ_VLLM_SPARSE_RECORD_COMPONENT_RETAIN_SCORE_ENV,
    AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV,
    SparsePrefillSelectionStats,
    SparsePrefillTopKConfig,
    _build_sparse_selection_component_keep_masks,
    _build_sparse_attention_mask_topk,
    _select_topk_sparse_blocks,
    _select_threshold_sparse_blocks,
    _reconstruct_sequence_slots,
    gather_full_sequence_kv_from_paged_cache,
    format_sparse_prefill_selection_policy,
    get_sparse_prefill_impl_mode,
    get_sparse_prefill_retain_score_log_mode,
    get_sparse_prefill_topk_config,
    is_sparse_prefill_component_retain_score_recording_enabled,
    is_sparse_prefill_retain_score_recording_enabled,
    is_cached_prefix_prefill_request,
    is_full_prefill_request,
    is_sparse_prefill_request,
    resolve_sparse_prefill_layer_info,
    run_sparse_prefill_attention,
)
from vllm.v1.attention.ops import triton_sparse_prefill as triton_sparse_prefill_module
from vllm.v1.attention.ops.triton_sparse_prefill import (
    _log_triton_retain_score_stats,
    build_sparse_topk_block_metadata,
    build_sparse_topk_block_metadata_from_paged_cache,
)


def _kept_token_positions(sparse_mask: torch.Tensor) -> list[list[int]]:
    if sparse_mask.shape[2] != 1:
        raise ValueError(
            "Test helper expects q_len == 1, got sparse mask shape "
            f"{tuple(sparse_mask.shape)}."
        )
    mask_floor = torch.finfo(sparse_mask.dtype).min / 2
    visible = sparse_mask[0, :, 0, :] > mask_floor
    return [row.nonzero(as_tuple=False).reshape(-1).tolist() for row in visible]


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


@pytest.mark.parametrize(
    ("env_value", "expected_enabled", "expected_mode"),
    [
        ("1", True, "head"),
        ("true", True, "head"),
        ("On", True, "head"),
        ("summary", True, "summary"),
        ("layer", True, "layer"),
        ("head", True, "head"),
        ("0", False, "off"),
        ("", False, "off"),
    ],
)
def test_sparse_prefill_retain_score_switch_parses_modes(
    monkeypatch, env_value: str, expected_enabled: bool, expected_mode: str
):
    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV, env_value)

    assert get_sparse_prefill_retain_score_log_mode() == expected_mode
    assert is_sparse_prefill_retain_score_recording_enabled() is expected_enabled


def test_sparse_prefill_retain_score_switch_defaults_to_off(monkeypatch):
    monkeypatch.delenv(AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV, raising=False)

    assert get_sparse_prefill_retain_score_log_mode() == "off"
    assert not is_sparse_prefill_retain_score_recording_enabled()


def test_sparse_prefill_retain_score_switch_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_RECORD_RETAIN_SCORE_ENV, "per_layer")

    with pytest.raises(ValueError, match="must be one of"):
        get_sparse_prefill_retain_score_log_mode()


@pytest.mark.parametrize(
    ("env_value", "expected_enabled"),
    [
        ("1", True),
        ("true", True),
        ("On", True),
        ("0", False),
        ("false", False),
        ("", False),
    ],
)
def test_sparse_prefill_component_retain_score_switch_parses_modes(
    monkeypatch, env_value: str, expected_enabled: bool
):
    monkeypatch.setenv(
        AUTOPTQ_VLLM_SPARSE_RECORD_COMPONENT_RETAIN_SCORE_ENV, env_value
    )

    assert (
        is_sparse_prefill_component_retain_score_recording_enabled()
        is expected_enabled
    )


def test_sparse_prefill_component_retain_score_switch_defaults_to_off(monkeypatch):
    monkeypatch.delenv(
        AUTOPTQ_VLLM_SPARSE_RECORD_COMPONENT_RETAIN_SCORE_ENV, raising=False
    )

    assert not is_sparse_prefill_component_retain_score_recording_enabled()


def test_sparse_prefill_component_retain_score_switch_rejects_invalid_mode(
    monkeypatch,
):
    monkeypatch.setenv(
        AUTOPTQ_VLLM_SPARSE_RECORD_COMPONENT_RETAIN_SCORE_ENV, "components"
    )

    with pytest.raises(ValueError, match="must be one of"):
        is_sparse_prefill_component_retain_score_recording_enabled()


def test_sparse_force_paged_full_prefill_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv(
        AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV, raising=False
    )

    assert (
        _parse_optional_env_flag(
            AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV,
            default=True,
        )
        is True
    )


@pytest.mark.parametrize("env_value", ["0", "false", "off", "No"])
def test_sparse_force_paged_full_prefill_accepts_explicit_disable(
    monkeypatch, env_value: str
):
    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV, env_value)

    assert (
        _parse_optional_env_flag(
            AUTOPTQ_VLLM_SPARSE_FORCE_PAGED_FULL_PREFILL_ENV,
            default=True,
        )
        is False
    )


def test_resolve_sparse_prefill_layer_info_extracts_layer_name_and_index():
    layer = SimpleNamespace(layer_name="model.layers.7.self_attn.attn")

    layer_idx, layer_name = resolve_sparse_prefill_layer_info(layer)

    assert layer_idx == 7
    assert layer_name == "model.layers.7.self_attn.attn"


def test_resolve_sparse_prefill_layer_info_handles_missing_layer_name():
    layer_idx, layer_name = resolve_sparse_prefill_layer_info(object())

    assert layer_idx is None
    assert layer_name is None


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
    assert cfg.threshold is None
    assert cfg.sink_block == 0
    assert cfg.sliding_window_block == 0
    assert cfg.gqa_shared == "none"
    assert cfg.max_selected_blocks == 128
    assert format_sparse_prefill_selection_policy(cfg) == "topk=128"


def test_get_sparse_prefill_topk_config_parses_sink_and_sliding_window_blocks(
    monkeypatch, tmp_path
):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_test_20": {
                    "name": "q_256_k_1_topk_2048_swa_4_128",
                    "q_block": 256,
                    "k_block": 1,
                    "topk": 2048,
                    "sink_block": 4,
                    "sliding_window_block": 128,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_test_20")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.key == "sparse_test_20"
    assert cfg.q_block == 256
    assert cfg.k_block == 1
    assert cfg.topk == 2048
    assert cfg.threshold is None
    assert cfg.sink_block == 4
    assert cfg.sliding_window_block == 128
    assert cfg.max_selected_blocks == 2180
    assert format_sparse_prefill_selection_policy(cfg) == (
        "topk=2048,sink_block=4,sliding_window_block=128"
    )


def test_get_sparse_prefill_topk_config_rejects_q_pooling_field(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_bad": {
                    "name": "bad_sparse",
                    "q_block": 256,
                    "k_block": 1,
                    "topk": 2048,
                    "q_pooling": "mean_after",
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_bad")

    with pytest.raises(ValueError, match="field 'q_pooling' is no longer supported"):
        get_sparse_prefill_topk_config()


def test_get_sparse_prefill_topk_config_parses_xattn_scheme(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_test_46": {
                    "name": "q_64_k_16_topk_128_xattn",
                    "q_block": 64,
                    "k_block": 16,
                    "topk": 128,
                    "xattn": True,
                    "xattn_stride": 8,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_test_46")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.xattn is True
    assert cfg.xattn_stride == 8
    assert format_sparse_prefill_selection_policy(cfg) == "topk=128,xattn_stride=8"


def test_get_sparse_prefill_topk_config_parses_k_sum_scheme(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_test_41_2": {
                    "name": "q_16_k_16_topk_128_k_sum",
                    "q_block": 16,
                    "k_block": 16,
                    "topk": 128,
                    "k_sum": True,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_test_41_2")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.k_sum is True
    assert cfg.xattn is False
    assert format_sparse_prefill_selection_policy(cfg) == "topk=128,k_sum=true"


def test_get_sparse_prefill_topk_config_parses_gqa_shared_mean(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_test_57": {
                    "name": "q_32_gqa_shared_k_16_threshold_0.87",
                    "q_block": 32,
                    "k_block": 16,
                    "threshold": 0.87,
                    "gqa_shared": "mean",
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_test_57")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.gqa_shared == "mean"
    assert format_sparse_prefill_selection_policy(cfg) == (
        "threshold=0.87,gqa_shared=mean"
    )


def test_get_sparse_prefill_topk_config_parses_gqa_shared_mean_before(
    monkeypatch, tmp_path
):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_test_63": {
                    "name": "q_32_gqa_shared_k_16_threshold_0.87_sink1_swa8",
                    "q_block": 32,
                    "k_block": 16,
                    "threshold": 0.87,
                    "gqa_shared": "mean_before",
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_test_63")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.gqa_shared == "mean_before"
    assert format_sparse_prefill_selection_policy(cfg) == (
        "threshold=0.87,gqa_shared=mean_before"
    )


def test_get_sparse_prefill_topk_config_parses_threshold_scheme(
    monkeypatch, tmp_path
):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_test_32": {
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
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_test_32")

    cfg = get_sparse_prefill_topk_config()
    assert cfg is not None
    assert cfg.topk is None
    assert cfg.threshold == pytest.approx(0.95)
    assert cfg.max_selected_blocks == 0
    assert format_sparse_prefill_selection_policy(cfg) == "threshold=0.95"


@pytest.mark.parametrize("field_name", ["sink_block", "sliding_window_block"])
def test_get_sparse_prefill_topk_config_rejects_negative_sink_or_window_blocks(
    monkeypatch, tmp_path, field_name: str
):
    sparse_json = tmp_path / "sparse.json"
    payload = {
        "sparse_bad": {
            "name": "bad_sparse",
            "q_block": 16,
            "k_block": 1,
            "topk": 32,
            "sink_block": 0,
            "sliding_window_block": 0,
        }
    }
    payload["sparse_bad"][field_name] = -1
    sparse_json.write_text(json.dumps(payload))

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_bad")

    with pytest.raises(ValueError, match=rf"Sparse scheme field '{field_name}'.*>= 0"):
        get_sparse_prefill_topk_config()


def test_get_sparse_prefill_topk_config_rejects_invalid_gqa_shared(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_bad": {
                    "name": "bad_sparse",
                    "q_block": 16,
                    "k_block": 1,
                    "topk": 32,
                    "gqa_shared": "sum",
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_bad")

    with pytest.raises(ValueError, match="field 'gqa_shared' must be one of"):
        get_sparse_prefill_topk_config()


def test_get_sparse_prefill_topk_config_rejects_xattn_stride_without_xattn(
    monkeypatch, tmp_path
):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_bad": {
                    "name": "bad_sparse",
                    "q_block": 64,
                    "k_block": 16,
                    "topk": 128,
                    "xattn_stride": 8,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_bad")

    with pytest.raises(ValueError, match="xattn_stride' requires 'xattn'=true"):
        get_sparse_prefill_topk_config()


def test_get_sparse_prefill_topk_config_rejects_nonbool_k_sum(monkeypatch, tmp_path):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_bad": {
                    "name": "bad_sparse",
                    "q_block": 16,
                    "k_block": 16,
                    "topk": 32,
                    "k_sum": "yes",
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_bad")

    with pytest.raises(ValueError, match="field 'k_sum' must be a boolean"):
        get_sparse_prefill_topk_config()


def test_sparse_prefill_topk_config_rejects_gqa_shared_mean_before_xattn():
    with pytest.raises(ValueError, match="gqa_shared'='mean_before'.*xattn'=true"):
        SparsePrefillTopKConfig(
            key="test_sparse",
            name="test_sparse",
            q_block=64,
            k_block=16,
            topk=128,
            gqa_shared="mean_before",
            xattn=True,
            xattn_stride=8,
        )


def test_sparse_prefill_topk_config_rejects_k_sum_with_xattn():
    with pytest.raises(ValueError, match="k_sum'=true.*xattn'=true"):
        SparsePrefillTopKConfig(
            key="test_sparse",
            name="test_sparse",
            q_block=64,
            k_block=16,
            topk=128,
            xattn=True,
            k_sum=True,
            xattn_stride=8,
        )


def test_sparse_prefill_topk_config_rejects_xattn_stride_not_dividing_blocks():
    with pytest.raises(ValueError, match="k_block=16 must be divisible"):
        SparsePrefillTopKConfig(
            key="test_sparse",
            name="test_sparse",
            q_block=60,
            k_block=16,
            topk=128,
            xattn=True,
            xattn_stride=6,
        )


def test_build_sparse_topk_block_metadata_records_retained_attention_score():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=2,
        topk=1,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    pooled_key = torch.tensor([[[[0.0], [1.0], [2.0]]]], dtype=torch.float32)

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=2.0,
        cfg=cfg,
        k_len=6,
        record_selection_stats=True,
    )

    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_valid_blocks == 5
    assert metadata.selection_stats.total_kept_blocks == 2
    assert metadata.selection_stats.total_valid_rows == 2
    assert metadata.selection_stats.per_head_total_valid_blocks == (5,)
    assert metadata.selection_stats.per_head_total_kept_blocks == (2,)
    assert metadata.selection_stats.per_head_total_valid_rows == (2,)

    row0 = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[-1]
    row1 = torch.softmax(torch.tensor([0.0, 2.0, 4.0]), dim=0)[-1]
    expected_mean = float((row0 + row1) / 2)
    assert metadata.selection_stats.retained_attention_score_mean == pytest.approx(
        expected_mean,
        rel=1e-6,
    )
    assert metadata.selection_stats.per_head_retained_attention_score_mean == (
        pytest.approx(expected_mean, rel=1e-6),
    )


def test_build_sparse_topk_block_metadata_records_per_head_retained_attention_score():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=2,
        topk=1,
    )
    query = torch.ones((4, 2, 1), dtype=torch.float32)
    pooled_key = torch.tensor(
        [[[[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]]]],
        dtype=torch.float32,
    )

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=1.0,
        cfg=cfg,
        k_len=6,
        record_selection_stats=True,
    )

    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_valid_blocks == 10
    assert metadata.selection_stats.total_kept_blocks == 4
    assert metadata.selection_stats.per_head_total_valid_blocks == (5, 5)
    assert metadata.selection_stats.per_head_total_kept_blocks == (2, 2)
    assert metadata.selection_stats.per_head_total_valid_rows == (2, 2)

    head0_row0 = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[-1]
    head0_row1 = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)[-1]
    head1_row0 = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[-1]
    head1_row1 = torch.softmax(torch.tensor([0.0, 2.0, 4.0]), dim=0)[-1]
    expected = (
        float((head0_row0 + head0_row1) / 2),
        float((head1_row0 + head1_row1) / 2),
    )
    assert metadata.selection_stats.per_head_retained_attention_score_mean == pytest.approx(
        expected,
        rel=1e-6,
    )


def test_build_sparse_topk_block_metadata_merges_sink_window_and_remainder_topk():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=1,
        topk=1,
        sink_block=1,
        sliding_window_block=1,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    pooled_key = torch.tensor(
        [[[[0.0], [1.0], [10.0], [20.0], [30.0], [40.0]]]],
        dtype=torch.float32,
    )

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=1.0,
        cfg=cfg,
        k_len=6,
        record_selection_stats=True,
    )

    assert metadata.topk_block_indices.shape == (1, 2, 3)
    assert metadata.topk_block_indices.shape[2] > cfg.topk
    assert metadata.topk_block_counts.tolist() == [[3, 3]]
    assert metadata.topk_block_indices[0, 0].tolist() == [0, 2, 3]
    assert metadata.topk_block_indices[0, 1].tolist() == [0, 4, 5]
    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_kept_blocks == 6
    assert metadata.component_selection_stats is not None
    assert metadata.component_selection_stats["sink_block"].total_kept_blocks == 2
    assert (
        metadata.component_selection_stats["sliding_window_block"].total_kept_blocks
        == 2
    )
    assert (
        metadata.component_selection_stats["sliding_window_block_0"].total_kept_blocks
        == 2
    )
    assert metadata.component_selection_stats["mandatory_union"].total_kept_blocks == 4


def test_build_sparse_selection_component_keep_masks_splits_sliding_window_blocks():
    valid_block_mask = torch.tensor(
        [
            [
                [
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, True],
                ]
            ]
        ],
        dtype=torch.bool,
    )

    component_keep_masks = _build_sparse_selection_component_keep_masks(
        valid_block_mask,
        sink_block=1,
        sliding_window_block=2,
    )

    assert list(component_keep_masks) == [
        "sink_block",
        "sliding_window_block",
        "sliding_window_block_0",
        "sliding_window_block_1",
        "mandatory_union",
    ]
    assert component_keep_masks["sliding_window_block"].tolist() == [
        [
            [
                [False, False, True, True, False, False],
                [False, False, False, False, True, True],
            ]
        ]
    ]
    assert component_keep_masks["sliding_window_block_0"].tolist() == [
        [
            [
                [False, False, False, True, False, False],
                [False, False, False, False, False, True],
            ]
        ]
    ]
    assert component_keep_masks["sliding_window_block_1"].tolist() == [
        [
            [
                [False, False, True, False, False, False],
                [False, False, False, False, True, False],
            ]
        ]
    ]
    assert component_keep_masks["mandatory_union"].tolist() == [
        [
            [
                [True, False, True, True, False, False],
                [True, False, False, False, True, True],
            ]
        ]
    ]


def test_build_sparse_topk_block_metadata_records_per_sliding_window_component_masks():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=1,
        topk=1,
        sink_block=1,
        sliding_window_block=2,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    pooled_key = torch.tensor(
        [[[[0.0], [1.0], [10.0], [20.0], [30.0], [40.0]]]],
        dtype=torch.float32,
    )

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=1.0,
        cfg=cfg,
        k_len=6,
        build_retain_tile_block_mask=True,
        record_selection_stats=True,
    )

    assert metadata.component_selection_stats is not None
    assert (
        metadata.component_selection_stats["sliding_window_block"].total_kept_blocks
        == 4
    )
    assert (
        metadata.component_selection_stats["sliding_window_block_0"].total_kept_blocks
        == 2
    )
    assert (
        metadata.component_selection_stats["sliding_window_block_1"].total_kept_blocks
        == 2
    )
    assert metadata.component_retain_tile_block_masks is not None
    assert "sliding_window_block_0" in metadata.component_retain_tile_block_masks
    assert "sliding_window_block_1" in metadata.component_retain_tile_block_masks


def test_build_sparse_topk_block_metadata_can_disable_component_stats():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=1,
        topk=1,
        sink_block=1,
        sliding_window_block=2,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    pooled_key = torch.tensor(
        [[[[0.0], [1.0], [10.0], [20.0], [30.0], [40.0]]]],
        dtype=torch.float32,
    )

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=1.0,
        cfg=cfg,
        k_len=6,
        build_retain_tile_block_mask=True,
        record_selection_stats=True,
        record_component_selection_stats=False,
    )

    assert metadata.selection_stats is not None
    assert metadata.component_selection_stats is None
    assert metadata.component_retain_tile_block_masks is None


def test_build_sparse_topk_block_metadata_dedupes_sink_window_overlap():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=1,
        topk=1,
        sink_block=2,
        sliding_window_block=2,
    )
    query = torch.ones((2, 1, 1), dtype=torch.float32)
    pooled_key = torch.tensor([[[[0.0], [1.0], [2.0]]]], dtype=torch.float32)

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=1.0,
        cfg=cfg,
        k_len=3,
        record_selection_stats=True,
    )

    assert metadata.topk_block_indices.shape == (1, 1, 3)
    assert metadata.topk_block_counts.tolist() == [[3]]
    assert metadata.topk_block_indices[0, 0].tolist() == [0, 1, 2]
    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_kept_blocks == 3


def test_select_topk_sparse_blocks_keeps_block_zero_when_counts_below_topk():
    block_values = torch.tensor([[[[9.0, 8.0, 7.0, 6.0, -1.0, -1.0]]]], dtype=torch.float32)
    valid_block_mask = torch.tensor(
        [[[[True, True, True, True, False, False]]]],
        dtype=torch.bool,
    )

    keep_mask = _select_topk_sparse_blocks(
        block_values,
        valid_block_mask,
        topk=6,
    )

    assert keep_mask.tolist() == [[[[True, True, True, True, False, False]]]]


def test_select_threshold_sparse_blocks_keeps_minimal_mass_prefix():
    block_probs = torch.tensor([[[[0.45, 0.30, 0.15, 0.10]]]], dtype=torch.float32)
    valid_block_mask = torch.ones_like(block_probs, dtype=torch.bool)

    keep_mask = _select_threshold_sparse_blocks(
        block_probs,
        valid_block_mask,
        required_mass=torch.tensor([[[[0.75]]]], dtype=torch.float32),
    )

    assert keep_mask.tolist() == [[[[True, True, False, False]]]]


def test_build_sparse_topk_block_metadata_threshold_accounts_for_mandatory_blocks():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=4,
        k_block=1,
        threshold=0.95,
        sink_block=1,
        sliding_window_block=1,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    pooled_key = torch.tensor([[[[0.0], [1.0], [2.0], [4.0]]]], dtype=torch.float32)

    metadata = build_sparse_topk_block_metadata(
        query=query,
        pooled_key=pooled_key,
        scaling=1.0,
        cfg=cfg,
        k_len=4,
        record_selection_stats=True,
    )

    assert metadata.topk_block_indices.shape == (1, 1, 3)
    assert metadata.topk_block_counts.tolist() == [[3]]
    assert metadata.topk_block_indices[0, 0].tolist() == [0, 2, 3]
    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_kept_blocks == 3


def test_build_sparse_attention_mask_topk_gqa_shared_mean_shares_topk_selection():
    query = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 2.0]]]],
        dtype=torch.float32,
    )
    base_key = torch.tensor(
        [[[[1.1, 0.0], [0.0, 1.0], [0.75, 0.75]]]],
        dtype=torch.float32,
    )
    key = base_key.expand(1, 2, -1, -1).clone()
    attention_mask = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    no_share_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        topk=1,
    )
    shared_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        topk=1,
        gqa_shared="mean",
    )

    no_share_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=no_share_cfg,
        num_kv_heads=1,
    )
    shared_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=shared_cfg,
        num_kv_heads=1,
    )

    assert _kept_token_positions(no_share_mask) == [[0], [1]]
    assert _kept_token_positions(shared_mask) == [[2], [2]]


def test_build_sparse_attention_mask_topk_k_sum_changes_topk_block_selection():
    query = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    key = torch.tensor([[[[2.0], [-2.0], [1.0]]]], dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    mean_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=2,
        topk=1,
    )
    k_sum_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=2,
        topk=1,
        k_sum=True,
    )

    mean_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=mean_cfg,
        num_kv_heads=1,
    )
    k_sum_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=k_sum_cfg,
        num_kv_heads=1,
    )

    assert _kept_token_positions(mean_mask) == [[2]]
    assert _kept_token_positions(k_sum_mask) == [[0, 1]]


def test_build_sparse_attention_mask_topk_k_sum_changes_threshold_selection():
    query = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    key = torch.tensor([[[[2.0], [-2.0], [1.0]]]], dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    mean_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=2,
        threshold=0.5,
    )
    k_sum_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=2,
        threshold=0.5,
        k_sum=True,
    )

    mean_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=mean_cfg,
        num_kv_heads=1,
    )
    k_sum_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=k_sum_cfg,
        num_kv_heads=1,
    )

    assert _kept_token_positions(mean_mask) == [[2]]
    assert _kept_token_positions(k_sum_mask) == [[0, 1]]


def test_build_sparse_attention_mask_topk_gqa_shared_mean_shares_threshold_selection():
    query = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 2.0]]]],
        dtype=torch.float32,
    )
    base_key = torch.tensor(
        [[[[1.1, 0.0], [0.0, 1.0], [0.75, 0.75]]]],
        dtype=torch.float32,
    )
    key = base_key.expand(1, 2, -1, -1).clone()
    attention_mask = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    no_share_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        threshold=0.55,
    )
    shared_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        threshold=0.55,
        gqa_shared="mean",
    )

    no_share_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=no_share_cfg,
        num_kv_heads=1,
    )
    shared_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=shared_cfg,
        num_kv_heads=1,
    )

    assert _kept_token_positions(no_share_mask) == [[0], [1]]
    assert _kept_token_positions(shared_mask) == [[0, 2], [0, 2]]


def test_build_sparse_attention_mask_topk_gqa_shared_mean_before_shares_before_score():
    query = torch.tensor(
        [[[[2.1, 0.0]], [[0.0, 2.0]]]],
        dtype=torch.float32,
    )
    base_key = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [0.8, 0.8]]]],
        dtype=torch.float32,
    )
    key = base_key.expand(1, 2, -1, -1).clone()
    attention_mask = torch.full((1, 2, 1, 3), torch.finfo(torch.float32).min)
    attention_mask[0, 0, 0, [0, 2]] = 0.0
    attention_mask[0, 1, 0, [1, 2]] = 0.0

    no_share_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        topk=1,
    )
    shared_after_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        topk=1,
        gqa_shared="mean",
    )
    shared_before_cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=1,
        k_block=1,
        topk=1,
        gqa_shared="mean_before",
    )

    no_share_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=no_share_cfg,
        num_kv_heads=1,
    )
    shared_after_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=shared_after_cfg,
        num_kv_heads=1,
    )
    shared_before_mask, _ = _build_sparse_attention_mask_topk(
        query,
        key,
        attention_mask,
        scaling=1.0,
        cfg=shared_before_cfg,
        num_kv_heads=1,
    )

    assert _kept_token_positions(no_share_mask) == [[0], [1]]
    assert _kept_token_positions(shared_after_mask) == [[0], []]
    assert _kept_token_positions(shared_before_mask) == [[2], [2]]


def test_build_sparse_topk_block_metadata_supports_xattn_asymmetric_blocks():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=4,
        k_block=2,
        topk=1,
        xattn=True,
        xattn_stride=2,
    )
    query = torch.ones((8, 1, 1), dtype=torch.float32)
    key = torch.tensor(
        [[[0.0]], [[0.0]], [[1.0]], [[1.0]], [[2.0]], [[2.0]], [[3.0]], [[3.0]]],
        dtype=torch.float32,
    )

    metadata = build_sparse_topk_block_metadata(
        query=query,
        key=key,
        scaling=1.0,
        cfg=cfg,
        k_len=8,
        record_selection_stats=True,
    )

    assert metadata.topk_block_indices.shape == (1, 2, 1)
    assert metadata.topk_block_counts.tolist() == [[1, 1]]
    assert metadata.topk_block_indices[0, 0, 0].item() == 1
    assert metadata.topk_block_indices[0, 1, 0].item() == 3
    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_valid_blocks == 6
    assert metadata.selection_stats.total_kept_blocks == 2


def test_build_sparse_topk_block_metadata_from_paged_cache_supports_xattn():
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=4,
        k_block=2,
        topk=1,
        xattn=True,
        xattn_stride=2,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    full_key = torch.tensor(
        [[[0.0]], [[0.0]], [[1.0]], [[1.0]], [[2.0]], [[2.0]], [[3.0]], [[3.0]]],
        dtype=torch.float32,
    )
    full_value = full_key.clone()
    block_table_row = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    kv_cache = torch.zeros((2, 4, 2, 1, 1), dtype=torch.float32)
    for logical_block in range(4):
        physical_block = int(block_table_row[logical_block].item())
        start = logical_block * 2
        end = start + 2
        kv_cache[0, physical_block, :2].copy_(full_key[start:end])
        kv_cache[1, physical_block, :2].copy_(full_value[start:end])

    metadata = build_sparse_topk_block_metadata_from_paged_cache(
        query=query,
        kv_cache=kv_cache,
        block_table_row=block_table_row,
        seq_len=8,
        scaling=1.0,
        kv_cache_dtype="auto",
        cfg=cfg,
        record_selection_stats=True,
    )

    assert metadata.topk_block_indices.shape == (1, 1, 1)
    assert metadata.topk_block_counts.tolist() == [[1]]
    assert metadata.topk_block_indices[0, 0, 0].item() == 3
    assert metadata.selection_stats is not None
    assert metadata.selection_stats.total_valid_blocks == 4
    assert metadata.selection_stats.total_kept_blocks == 1


@pytest.mark.parametrize(
    ("retain_score_log_mode", "expect_layer", "expect_per_head"),
    [
        ("summary", False, False),
        ("layer", True, False),
        ("head", True, True),
        (None, True, True),
    ],
)
def test_run_sparse_prefill_attention_logs_retain_score_by_mode(
    monkeypatch,
    retain_score_log_mode: str | None,
    expect_layer: bool,
    expect_per_head: bool,
):
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=2,
        topk=1,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    key = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], dtype=torch.float32)
    value = key.clone()
    layer = SimpleNamespace(layer_name="model.layers.3.self_attn.attn")

    logged: list[str] = []

    def _capture_info(message: str, *args):
        logged.append(message % args)

    monkeypatch.setattr(sparse_prefill_utils_module.logger, "info", _capture_info)

    run_sparse_prefill_attention(
        query=query,
        key=key,
        value=value,
        scaling=1.0,
        cfg=cfg,
        output_dtype=torch.float32,
        record_retain_score=True,
        retain_score_log_mode=retain_score_log_mode,
        layer=layer,
    )

    summary_messages = [
        message
        for message in logged
        if message.startswith("Sparse prefill torch retain-score stats:")
    ]
    per_head_messages = [
        message
        for message in logged
        if message.startswith("Sparse prefill torch retain-score per-head stats:")
    ]

    assert len(summary_messages) == 1
    if expect_layer:
        assert (
            "layer_idx=3 layer_name=model.layers.3.self_attn.attn"
            in summary_messages[0]
        )
    else:
        assert "layer_idx=" not in summary_messages[0]
        assert "layer_name=" not in summary_messages[0]

    if expect_per_head:
        assert len(per_head_messages) == 1
        assert (
            "retain_scores=[0.940399]" in per_head_messages[0]
            and "densities=[0.666667]" in per_head_messages[0]
            and "valid_rows=[2]" in per_head_messages[0]
            and "topk=1" in per_head_messages[0]
        )
    else:
        assert not per_head_messages
    assert "topk=1" in summary_messages[0]


def test_run_sparse_prefill_attention_logs_threshold_policy(monkeypatch):
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=2,
        k_block=2,
        threshold=0.95,
    )
    query = torch.ones((4, 1, 1), dtype=torch.float32)
    key = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], dtype=torch.float32)
    value = key.clone()

    logged: list[str] = []

    def _capture_info(message: str, *args):
        logged.append(message % args)

    monkeypatch.setattr(sparse_prefill_utils_module.logger, "info", _capture_info)

    run_sparse_prefill_attention(
        query=query,
        key=key,
        value=value,
        scaling=1.0,
        cfg=cfg,
        output_dtype=torch.float32,
        record_retain_score=True,
        retain_score_log_mode="head",
    )

    summary_messages = [
        message
        for message in logged
        if message.startswith("Sparse prefill torch retain-score stats:")
    ]
    per_head_messages = [
        message
        for message in logged
        if message.startswith("Sparse prefill torch retain-score per-head stats:")
    ]

    assert len(summary_messages) == 1
    assert len(per_head_messages) == 1
    assert "threshold=0.95" in summary_messages[0]
    assert "threshold=0.95" in per_head_messages[0]
    assert "topk=" not in summary_messages[0]
    assert "topk=" not in per_head_messages[0]


def test_log_triton_retain_score_stats_logs_approx_and_exact(monkeypatch):
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=16,
        k_block=16,
        threshold=0.9,
    )
    approx_stats = SparsePrefillSelectionStats(
        total_valid_blocks=10,
        total_kept_blocks=5,
        total_valid_rows=2,
        retained_attention_score_sum=1.8,
    )
    exact_stats = SparsePrefillSelectionStats(
        total_valid_blocks=10,
        total_kept_blocks=5,
        total_valid_rows=2,
        retained_attention_score_sum=1.96,
    )

    logged: list[str] = []

    def _capture_info(message: str, *args):
        logged.append(message % args)

    monkeypatch.setattr(triton_sparse_prefill_module.logger, "info", _capture_info)

    _log_triton_retain_score_stats(
        layer=None,
        query_len=32,
        seq_len=64,
        cfg=cfg,
        paged_kv=True,
        selection_stats=exact_stats,
        approx_selection_stats=approx_stats,
        retain_score_log_mode="summary",
    )

    assert len(logged) == 1
    assert "threshold=0.9" in logged[0]
    assert "approx_retain_mass=0.900000" in logged[0]
    assert "avg_retain_score=0.980000" in logged[0]


def test_log_triton_retain_score_stats_includes_component_name(monkeypatch):
    cfg = SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=16,
        k_block=16,
        topk=4,
        sink_block=1,
        sliding_window_block=2,
    )
    stats = SparsePrefillSelectionStats(
        total_valid_blocks=32,
        total_kept_blocks=4,
        total_valid_rows=8,
        retained_attention_score_sum=7.2,
    )

    logged: list[str] = []

    def _capture_info(message: str, *args):
        logged.append(message % args)

    monkeypatch.setattr(triton_sparse_prefill_module.logger, "info", _capture_info)

    _log_triton_retain_score_stats(
        layer=None,
        query_len=32,
        seq_len=64,
        cfg=cfg,
        paged_kv=True,
        selection_stats=stats,
        approx_selection_stats=stats,
        component_name="sink_block",
        retain_score_log_mode="summary",
    )

    assert len(logged) == 1
    assert "component=sink_block" in logged[0]
    assert "topk=4,sink_block=1,sliding_window_block=2" in logged[0]


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


def test_get_sparse_prefill_topk_config_rejects_missing_selection_policy(
    monkeypatch, tmp_path
):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_v1": {
                    "name": "q_16_k_16_invalid",
                    "q_block": 16,
                    "k_block": 16,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_v1")

    with pytest.raises(ValueError, match="must define exactly one of 'topk' or 'threshold'"):
        get_sparse_prefill_topk_config()


def test_get_sparse_prefill_topk_config_rejects_multiple_selection_policies(
    monkeypatch, tmp_path
):
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(
        json.dumps(
            {
                "sparse_v1": {
                    "name": "q_16_k_16_invalid",
                    "q_block": 16,
                    "k_block": 16,
                    "topk": 32,
                    "threshold": 0.95,
                }
            }
        )
    )

    monkeypatch.setenv(AUTOPTQ_VLLM_SPARSE_ENABLE_ENV, "1")
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_JSON_ENV, str(sparse_json))
    monkeypatch.setenv(AUTOPTQ_SPARSE_RUNTIME_KEY_ENV, "sparse_v1")

    with pytest.raises(ValueError, match="must define exactly one of 'topk' or 'threshold'"):
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

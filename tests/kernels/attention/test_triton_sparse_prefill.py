import math

import pytest
import torch

from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.sparse_prefill_utils import (
    SparsePrefillTopKConfig,
    run_sparse_prefill_attention,
)
from vllm.v1.attention.ops.triton_sparse_prefill import (
    _compute_paged_retain_scores,
    build_sparse_topk_block_metadata_from_paged_cache,
    run_triton_sparse_prefill_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)


def _make_cfg(*, topk: int = 4, q_block: int = 16, k_block: int = 16) -> SparsePrefillTopKConfig:
    return SparsePrefillTopKConfig(
        key="test_sparse",
        name="test_sparse",
        q_block=q_block,
        k_block=k_block,
        topk=topk,
    )


def _build_paged_kv_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table_row: torch.Tensor,
    cache_block_size: int,
) -> torch.Tensor:
    seq_len, num_kv_heads, head_dim = key.shape
    num_logical_blocks = (seq_len + cache_block_size - 1) // cache_block_size
    num_blocks = int(block_table_row.max().item()) + 1
    kv_cache = torch.zeros(
        (2, num_blocks, cache_block_size, num_kv_heads, head_dim),
        dtype=key.dtype,
        device=key.device,
    )
    for logical_block in range(num_logical_blocks):
        physical_block = int(block_table_row[logical_block].item())
        start = logical_block * cache_block_size
        end = min(start + cache_block_size, seq_len)
        token_count = end - start
        kv_cache[0, physical_block, :token_count].copy_(key[start:end])
        kv_cache[1, physical_block, :token_count].copy_(value[start:end])
    return kv_cache


def _repeat_kv_heads_for_reference(key: torch.Tensor, num_heads: int) -> torch.Tensor:
    num_kv_heads = key.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Expected num_heads={num_heads} to be divisible by num_kv_heads={num_kv_heads}."
        )
    repeats = num_heads // num_kv_heads
    return key if repeats == 1 else key.repeat_interleave(repeats, dim=1)


def _compute_paged_retain_scores_reference(
    *,
    query: torch.Tensor,
    full_key: torch.Tensor,
    scaling: float,
    cfg: SparsePrefillTopKConfig,
    topk_metadata,
) -> torch.Tensor:
    query_f32 = query.to(torch.float32)
    key_f32 = _repeat_kv_heads_for_reference(full_key.to(torch.float32), query.shape[1])
    seq_len = full_key.shape[0]
    q_len, num_heads, _ = query.shape
    q_abs_offset = int(topk_metadata.q_abs_offset)
    token_positions = torch.arange(seq_len, device=query.device, dtype=torch.int64)
    expected = torch.empty((q_len, num_heads), device=query.device, dtype=torch.float32)

    for q_idx in range(q_len):
        q_block_idx = q_idx // int(cfg.q_block)
        q_abs = q_idx + q_abs_offset
        causal_mask = token_positions <= q_abs

        for head_idx in range(num_heads):
            scores = torch.matmul(key_f32[:, head_idx], query_f32[q_idx, head_idx])
            masked_scores = torch.where(
                causal_mask,
                scores * float(scaling),
                torch.full((seq_len,), float("-inf"), device=query.device),
            )
            probs = torch.softmax(masked_scores, dim=0)

            count = int(topk_metadata.topk_block_counts[head_idx, q_block_idx].item())
            selected_blocks = topk_metadata.topk_block_indices[
                head_idx, q_block_idx, :count
            ]
            selected_mask = torch.zeros(seq_len, device=query.device, dtype=torch.bool)
            for block_idx in selected_blocks.tolist():
                if block_idx < 0:
                    continue
                block_start = int(block_idx) * int(cfg.k_block)
                block_end = min(block_start + int(cfg.k_block), seq_len)
                selected_mask[block_start:block_end] = True

            expected[q_idx, head_idx] = probs[causal_mask & selected_mask].sum()

    return expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [8, 2])
@pytest.mark.parametrize("k_block", [1, 4, 8, 16])
def test_triton_sparse_prefill_matches_pytorch_full_prefill(
    dtype: torch.dtype,
    num_kv_heads: int,
    k_block: int,
):
    torch.manual_seed(0)
    q_len = 130
    num_heads = 8
    head_dim = 64
    cfg = _make_cfg(topk=4, k_block=k_block)
    scale = 1.0 / math.sqrt(head_dim)

    query = torch.randn(q_len, num_heads, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(q_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn(q_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)

    reference = run_sparse_prefill_attention(
        query=query,
        key=key,
        value=value,
        scaling=scale,
        cfg=cfg,
        output_dtype=dtype,
    )
    output = run_triton_sparse_prefill_attention(
        query=query,
        key=key,
        value=value,
        scaling=scale,
        cfg=cfg,
        output_dtype=dtype,
    )

    atol = 4e-2 if dtype == torch.bfloat16 else 3e-3
    rtol = 4e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k_block", [1, 4, 8, 16])
def test_triton_sparse_prefill_matches_pytorch_cached_prefix(
    dtype: torch.dtype,
    k_block: int,
):
    torch.manual_seed(1)
    seq_len = 97
    query_len = 33
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    cache_block_size = 16
    cfg = _make_cfg(topk=4, k_block=k_block)
    scale = 1.0 / math.sqrt(head_dim)

    query = torch.randn(query_len, num_heads, head_dim, device="cuda", dtype=dtype)
    full_key = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    full_value = torch.randn(
        seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    block_table_row = torch.tensor(
        [3, 0, 5, 1, 6, 4, 2],
        device="cuda",
        dtype=torch.int32,
    )
    kv_cache = _build_paged_kv_cache(
        key=full_key,
        value=full_value,
        block_table_row=block_table_row,
        cache_block_size=cache_block_size,
    )

    reference = run_sparse_prefill_attention(
        query=query,
        key=full_key,
        value=full_value,
        scaling=scale,
        cfg=cfg,
        output_dtype=dtype,
    )
    output = run_triton_sparse_prefill_attention(
        query=query,
        kv_cache=kv_cache,
        block_table_row=block_table_row,
        seq_len=seq_len,
        kv_cache_dtype="auto",
        scaling=scale,
        cfg=cfg,
        output_dtype=dtype,
    )

    atol = 4e-2 if dtype == torch.bfloat16 else 3e-3
    rtol = 4e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k_block", [1, 4])
def test_triton_paged_retain_scores_match_pytorch_reference(
    dtype: torch.dtype,
    k_block: int,
):
    torch.manual_seed(3)
    seq_len = 37
    query_len = 11
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    cache_block_size = 8
    cfg = _make_cfg(topk=3, q_block=4, k_block=k_block)
    scale = 1.0 / math.sqrt(head_dim)

    query = torch.randn(query_len, num_heads, head_dim, device="cuda", dtype=dtype)
    full_key = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    full_value = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    block_table_row = torch.tensor(
        [2, 0, 4, 1, 3],
        device="cuda",
        dtype=torch.int32,
    )
    kv_cache = _build_paged_kv_cache(
        key=full_key,
        value=full_value,
        block_table_row=block_table_row,
        cache_block_size=cache_block_size,
    )
    topk_metadata = build_sparse_topk_block_metadata_from_paged_cache(
        query=query,
        kv_cache=kv_cache,
        block_table_row=block_table_row,
        seq_len=seq_len,
        scaling=scale,
        kv_cache_dtype="auto",
        cfg=cfg,
        build_retain_tile_block_mask=True,
        record_selection_stats=False,
    )

    actual = _compute_paged_retain_scores(
        query=query,
        kv_cache=kv_cache,
        block_table_row=block_table_row,
        seq_len=seq_len,
        kv_cache_dtype="auto",
        topk_metadata=topk_metadata,
        scaling=scale,
        cfg=cfg,
    )
    expected = _compute_paged_retain_scores_reference(
        query=query,
        full_key=full_key,
        scaling=scale,
        cfg=cfg,
        topk_metadata=topk_metadata,
    )

    assert topk_metadata.retain_tile_block_mask is not None
    assert torch.any(expected < 0.999)
    atol = 5e-3 if dtype == torch.bfloat16 else 3e-4
    rtol = 5e-3 if dtype == torch.bfloat16 else 3e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.benchmark
def test_triton_sparse_prefill_benchmark_smoke():
    torch.manual_seed(2)
    seq_len = 1024
    query_len = 256
    num_heads = 16
    num_kv_heads = 4
    head_dim = 64
    cache_block_size = 16
    cfg = _make_cfg(topk=16, k_block=1)
    scale = 1.0 / math.sqrt(head_dim)

    query = torch.randn(
        query_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_value = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_table_row = torch.randperm(
        (seq_len + cache_block_size - 1) // cache_block_size,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    kv_cache = _build_paged_kv_cache(
        key=full_key,
        value=full_value,
        block_table_row=block_table_row,
        cache_block_size=cache_block_size,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = run_triton_sparse_prefill_attention(
        query=query,
        kv_cache=kv_cache,
        block_table_row=block_table_row,
        seq_len=seq_len,
        kv_cache_dtype="auto",
        scaling=scale,
        cfg=cfg,
        output_dtype=query.dtype,
    )
    end.record()
    torch.cuda.synchronize()

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert start.elapsed_time(end) >= 0.0

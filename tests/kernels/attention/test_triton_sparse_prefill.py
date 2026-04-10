import math

import pytest
import torch

from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.sparse_prefill_utils import (
    SparsePrefillTopKConfig,
    run_sparse_prefill_attention,
)
from vllm.v1.attention.ops.triton_sparse_prefill import (
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

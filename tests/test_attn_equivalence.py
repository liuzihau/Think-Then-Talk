"""Bit-equivalence tests: dense additive bias path vs. FlexAttention BlockMask path.

These are the validation gate for axis 2 of the H200 refactor (T3_h200_refactor/02_flex_attention.md).

Run from the Think-Then-Talk repo root:
    pytest tests/test_attn_equivalence.py -v
"""
import pytest
import torch

from train.data_process import build_block_attention_mask
from model.attention.flex_block_mask import (
    build_t3_block_mask,
    make_t3_mask_mod,
    materialize_mask_mod,
)
from torch.nn.attention.flex_attention import flex_attention


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FlexAttention equivalence tests")


@pytest.mark.parametrize("inp_len,prefix_len,window_len,mask_len,block_size", [
    (16, 8,  8,  16, 2),
    (32, 16, 16, 32, 2),
    (8,  16, 16, 8,  4),
    (24, 0,  16, 16, 4),       # zero-prefix sample (start-of-target case)
])
def test_mask_pattern_equivalence(inp_len, prefix_len, window_len, mask_len, block_size):
    """The boolean mask materialized from mask_mod must match build_block_attention_mask."""
    _need_cuda()
    device = torch.device("cuda")
    total = inp_len + prefix_len + window_len + mask_len

    dense = build_block_attention_mask(
        max_length=total,
        inp_len=inp_len,
        prefix_len=prefix_len,
        window_len=window_len,
        mask_len=mask_len,
        block_size=block_size,
        device=device,
    )  # [S, S] bool, True = allow

    mask_mod = make_t3_mask_mod(
        inp_len   =torch.tensor([inp_len],    dtype=torch.int32, device=device),
        prefix_len=torch.tensor([prefix_len], dtype=torch.int32, device=device),
        window_len=torch.tensor([window_len], dtype=torch.int32, device=device),
        mask_len  =torch.tensor([mask_len],   dtype=torch.int32, device=device),
        block_size=block_size,
    )
    materialized = materialize_mask_mod(mask_mod, seq_len=total, device=device, batch_idx=0)

    if not torch.equal(dense, materialized):
        diff = (dense != materialized).nonzero(as_tuple=False)[:20]
        raise AssertionError(
            f"Mask pattern mismatch at {diff.shape[0]} positions; first 20: {diff.tolist()}"
        )


def test_attention_output_equivalence():
    """FlexAttention with BlockMask matches SDPA with dense bias to FP32 tolerance."""
    _need_cuda()
    device = torch.device("cuda")
    torch.manual_seed(0)

    B, H, D = 1, 4, 64
    inp_len, prefix_len, window_len, mask_len, block_size = 16, 8, 8, 16, 2
    S = inp_len + prefix_len + window_len + mask_len

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

    # Path A: dense bias + SDPA
    allow = build_block_attention_mask(
        max_length=S, inp_len=inp_len, prefix_len=prefix_len,
        window_len=window_len, mask_len=mask_len,
        block_size=block_size, device=device,
    )
    bias = torch.where(
        allow,
        torch.zeros_like(allow, dtype=torch.float32),
        torch.full_like(allow, float("-inf"), dtype=torch.float32),
    )
    bias = bias.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
    out_dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    # Path B: BlockMask + FlexAttention
    bm = build_t3_block_mask(
        inp_len   =torch.tensor([inp_len],    dtype=torch.int32, device=device),
        prefix_len=torch.tensor([prefix_len], dtype=torch.int32, device=device),
        window_len=torch.tensor([window_len], dtype=torch.int32, device=device),
        mask_len  =torch.tensor([mask_len],   dtype=torch.int32, device=device),
        block_size=block_size,
        seq_len=S,
        device=device,
    )
    out_flex = flex_attention(q, k, v, block_mask=bm)

    torch.testing.assert_close(out_dense, out_flex, atol=1e-5, rtol=1e-4)

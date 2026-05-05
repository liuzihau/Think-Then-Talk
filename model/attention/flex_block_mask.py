"""FlexAttention block mask for the T3 4-region layout.

Layout per sample:  [ inp | prefix | window | mask ]   padded to max_length.

Semantics:
  inp    = the static prompt the model conditions on.
  prefix = blocks already decoded in earlier denoise rounds (block-diffusion
           causality applies among them).
  window = ground-truth tokens for the k blocks being denoised this batch.
           window[b] is the label for mask[b]. Window blocks act as already-
           revealed context for later mask blocks (window[0] → mask[1+], etc.).
  mask   = placeholder slots being denoised now (training queries; loss is
           computed only here).

Allow rules (transcribed from build_block_attention_mask in
train/data_process.py:870-956, which is the source of truth):

    inp     -> inp                                   (bidirectional)
    prefix  -> inp ∪ prefix[0..b]                    (blockwise causal)
    window  -> inp ∪ prefix ∪ window[0..b]           (blockwise causal,
                                                      same pattern as prefix)
    mask    -> inp ∪ prefix ∪ window[0..b-1] ∪ mask[b..b]
               (sees all prior window blocks as revealed context, plus its
                own block; does NOT see its own label window[b] or any
                later window block)

Padded positions (q_idx >= total_len or kv_idx >= total_len) attend to nothing.

This file does NOT replace build_block_attention_mask; it produces the equivalent
mask_mod for FlexAttention. The dense version stays as the source of truth for
tests and visualization in train/visualize.py.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def make_t3_mask_mod(
    inp_len:    torch.Tensor,   # int [B]
    prefix_len: torch.Tensor,   # int [B]
    window_len: torch.Tensor,   # int [B]
    mask_len:   torch.Tensor,   # int [B]
    block_size: int,
):
    """Build a mask_mod closure for FlexAttention from per-sample region lengths.

    All length tensors must be on the same device as the q/k/v passed to
    flex_attention. The closure captures them by reference.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        ip = inp_len[b]
        pf = prefix_len[b]
        wn = window_len[b]
        mk = mask_len[b]

        o_inp = torch.zeros((), dtype=ip.dtype, device=ip.device)
        o_pre = o_inp + ip
        o_win = o_pre + pf
        o_msk = o_win + wn
        total = o_msk + mk

        in_bounds = (q_idx < total) & (kv_idx < total)

        # region membership of q
        q_in_inp = (q_idx >= o_inp) & (q_idx < o_pre)
        q_in_pre = (q_idx >= o_pre) & (q_idx < o_win)
        q_in_win = (q_idx >= o_win) & (q_idx < o_msk)
        q_in_msk = (q_idx >= o_msk) & (q_idx < total)

        # block index of q within its region (only meaningful in the corresponding region)
        q_pre_block = (q_idx - o_pre) // block_size
        q_win_block = (q_idx - o_win) // block_size
        q_msk_block = (q_idx - o_msk) // block_size

        # 1) inp -> inp (bidirectional within inp; inp does NOT see prefix/window/mask)
        allow_inp_to_inp = q_in_inp & (kv_idx >= o_inp) & (kv_idx < o_pre)

        # 2) prefix block b -> inp ∪ prefix[0..b]
        allow_pre_to_inp = q_in_pre & (kv_idx >= o_inp) & (kv_idx < o_pre)
        allow_pre_to_pre = (
            q_in_pre
            & (kv_idx >= o_pre)
            & (kv_idx < o_pre + (q_pre_block + 1) * block_size)
        )

        # 3) window block b -> inp ∪ prefix ∪ window[0..b]
        allow_win_to_inp = q_in_win & (kv_idx >= o_inp) & (kv_idx < o_pre)
        allow_win_to_pre = q_in_win & (kv_idx >= o_pre) & (kv_idx < o_win)
        allow_win_to_win = (
            q_in_win
            & (kv_idx >= o_win)
            & (kv_idx < o_win + (q_win_block + 1) * block_size)
        )

        # 4) mask block b -> inp ∪ prefix ∪ window[0..b-1] ∪ mask[b..b]
        allow_msk_to_inp = q_in_msk & (kv_idx >= o_inp) & (kv_idx < o_pre)
        allow_msk_to_pre = q_in_msk & (kv_idx >= o_pre) & (kv_idx < o_win)
        allow_msk_to_win = (
            q_in_msk
            & (kv_idx >= o_win)
            & (kv_idx < o_win + q_msk_block * block_size)
        )
        allow_msk_to_msk = (
            q_in_msk
            & (kv_idx >= o_msk + q_msk_block * block_size)
            & (kv_idx < o_msk + (q_msk_block + 1) * block_size)
        )

        allow = (
            allow_inp_to_inp
            | allow_pre_to_inp | allow_pre_to_pre
            | allow_win_to_inp | allow_win_to_pre | allow_win_to_win
            | allow_msk_to_inp | allow_msk_to_pre | allow_msk_to_win | allow_msk_to_msk
        )
        return allow & in_bounds

    return mask_mod


def build_t3_block_mask(
    inp_len:    torch.Tensor,
    prefix_len: torch.Tensor,
    window_len: torch.Tensor,
    mask_len:   torch.Tensor,
    block_size: int,
    seq_len:    int,
    device:     torch.device,
) -> BlockMask:
    """Build a FlexAttention BlockMask for one batch.

    Returns a BlockMask of logical shape (B, 1, seq_len, seq_len). The H dim is
    broadcast (None); the B dim is per-sample because each sample has its own
    region offsets.
    """
    B = int(inp_len.shape[0])
    mask_mod = make_t3_mask_mod(
        inp_len.to(device=device, dtype=torch.int32),
        prefix_len.to(device=device, dtype=torch.int32),
        window_len.to(device=device, dtype=torch.int32),
        mask_len.to(device=device, dtype=torch.int32),
        block_size=block_size,
    )
    return create_block_mask(
        mask_mod,
        B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device, _compile=True,
    )


def materialize_mask_mod(
    mask_mod,
    seq_len: int,
    device: torch.device,
    batch_idx: int = 0,
) -> torch.Tensor:
    """Materialize a mask_mod closure into a [seq_len, seq_len] bool tensor.

    Used only by tests / visualization. The returned tensor matches
    build_block_attention_mask's dense allow pattern when the closure is built
    from the same region offsets.
    """
    qs = torch.arange(seq_len, device=device).view(seq_len, 1).expand(seq_len, seq_len)
    ks = torch.arange(seq_len, device=device).view(1, seq_len).expand(seq_len, seq_len)
    b = torch.full((), batch_idx, dtype=torch.long, device=device)
    h = torch.zeros((), dtype=torch.long, device=device)
    return mask_mod(b, h, qs, ks)

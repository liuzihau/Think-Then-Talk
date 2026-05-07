"""Block-by-block decoding engine for T3 inference and eval.

Single source of truth for the per-prompt decode loop. Both `inference.py`
(interactive / debug entry point) and `eval_t3.py` (lm_eval harness) wrap
this engine; only the surrounding I/O differs.

## Decode flow (per block)

    1. Think forward over (cached prefix + current mask block) under a
       block-causal `attention_bias`: prompt is bidirectional within itself,
       each generated block sees prompt + already-decoded blocks + itself,
       and crucially each row cannot see *future* mask tokens. This matches
       the pre-axis-3 baseline (`f25cc1e`) and avoids the K/V pollution
       chain that the Fast-dLLM-style no-mask path suffers from on this
       checkpoint.
    2. Trim the trailing `block_size` entries from the KV cache so the next
       iteration recomputes K/V for those positions against the freshly
       decoded prefix.
    3. Slice the per-block talk inputs (input_ids + hidden representation)
       from the think output.
    4. Run the talk-side denoise loop for `steps_per_block` iterations,
       revealing tokens via the configured policy.
    5. Write the decoded tokens back into `x` and slide the window.

Caller decides tokenization, chat templating, prompt prefix/suffix, and
output decoding. Engine is stateless across `generate()` calls.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from utils import (
    denoise_k_step_hard,
    denoise_k_step_soft_embed_v2,
    get_denoise_decode_config,
    get_denoise_reveal_config,
)


# LLaDA mask token id. Hardcoded today because the supported backbones
# (LLaDA-8B-Instruct + the Qwen3 path that doesn't go through this engine)
# share this id. If a future backbone differs, look it up from tokenizer
# special tokens at engine construction.
LLADA_MASK_TOKEN_ID = 126336


class T3DecodeEngine:
    """Block-by-block T3 decoder. One engine per (model, config) pair.

    Construct once, call `generate(input_ids)` per prompt.
    """

    def __init__(
        self,
        model,
        model_config: dict,
        *,
        block_size: int,
        gen_length: int,
        mask_token_id: int = LLADA_MASK_TOKEN_ID,
        steps_per_block: int | None = None,
    ):
        # The old t3_generate silently floored num_blocks; we match that to
        # avoid a behaviour regression (e.g. eval_t3.sh defaults gen_length=512
        # with block_size=6 → 2 mask tokens trail). Warn instead of raising.
        if gen_length % block_size != 0:
            import warnings
            warnings.warn(
                f"gen_length ({gen_length}) is not a multiple of block_size "
                f"({block_size}); the trailing {gen_length % block_size} "
                "position(s) will remain as mask_token_id and not be decoded.",
                stacklevel=2,
            )
        self.model = model
        self.model_config = model_config
        self.block_size = block_size
        self.gen_length = gen_length
        self.num_blocks = gen_length // block_size
        self.mask_token_id = mask_token_id

        # Denoise schedule: pulled from model_config["denoise"], same recipe
        # as training. `steps_per_block` defaults to model.length (the talk
        # block size) but is overridable for ablation.
        denoise_cfg = model_config.get("denoise", {}) or {}
        self.reveal_cfg = get_denoise_reveal_config(denoise_cfg)
        self.decode_cfg = get_denoise_decode_config(
            denoise_cfg, default_k=self.reveal_cfg["k"]
        )
        self.reveal_k = int(self.reveal_cfg["k"])
        self.steps_per_block = (
            int(steps_per_block)
            if steps_per_block is not None
            else int(denoise_cfg.get("steps", model.length))
        )

        # soft_inputs is read once: the dict structure is config, not state.
        soft_cfg = model_config.get("soft_inputs", {}) or {}
        self.soft_inputs_enabled = bool(soft_cfg.get("enabled", False))
        self.soft_topk = int(soft_cfg.get("top_k", 0))
        self.soft_temp = float(soft_cfg.get("temperature", 1.0))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Generate `gen_length` tokens conditioned on `input_ids`.

        Args:
            input_ids: `[1, prompt_len]` long tensor on the model's device.
                Batch size > 1 is not supported (block-diffusion decode is
                inherently per-prompt).

        Returns:
            `(x, nfe)` where `x` is `[1, prompt_len + gen_length]` and `nfe`
            counts forward passes (one think + `steps_per_block` talk per block,
            stopping early once a block's loss_mask hits zero).
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"T3DecodeEngine.generate expects batch size 1, got {input_ids.shape[0]}"
            )

        device = input_ids.device
        seq_len = input_ids.shape[1]
        max_len = seq_len + self.gen_length

        x = torch.full((1, max_len), self.mask_token_id, dtype=torch.long, device=device)
        x[:, :seq_len] = input_ids
        position_ids = torch.arange(0, max_len, device=device).unsqueeze(0)
        attention_mask = torch.ones((1, max_len), dtype=torch.bool, device=device)
        attention_bias = _build_blockwise_attention_bias(
            seq_len=seq_len, max_len=max_len, block_size=self.block_size, device=device,
        )

        past_key_values = None
        cur_len = seq_len + self.block_size
        x0 = x[:, :cur_len]
        p0 = position_ids[:, :cur_len]

        nfe = 0
        for block_idx in range(self.num_blocks):
            think_outputs = self.model(
                input_ids=x0,
                attention_mask=attention_mask[:, :cur_len],
                attention_bias=attention_bias[:, :, :cur_len, :cur_len],
                position_ids=p0,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            nfe += 1

            past_key_values = _trim_kv_cache_trailing(
                think_outputs.past_key_values, n=self.block_size,
            )

            block_start = seq_len + self.block_size * block_idx
            block_end = block_start + self.block_size
            talk_input_ids, talk_rps = _select_talk_inputs(
                x0, think_outputs.hidden_states,
                block_size=self.block_size,
                block_start=block_start, block_end=block_end,
            )
            decoded_ids, talk_nfe = self._denoise_one_block(talk_input_ids, talk_rps)
            nfe += talk_nfe
            x[:, block_start:block_end] = decoded_ids

            # Slide the window for the next block.
            cur_len = block_end + self.block_size
            x0 = x[:, block_start:cur_len]
            p0 = position_ids[:, block_start:cur_len]

        return x, nfe

    def _denoise_one_block(
        self,
        talk_input_ids: torch.Tensor,
        talk_rps: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Run the talk-side denoise iterations for one block. Returns
        (final input_ids, nfe). Early-exits when the loss mask is all-zero."""
        loss_mask = torch.ones_like(talk_input_ids, dtype=torch.float32)
        talk_attn_mask = torch.ones_like(talk_input_ids, dtype=torch.long)
        talk_input_embeds = F.embedding(talk_input_ids, self.model.talk_embed_weight)

        nfe = 0
        for _ in range(self.steps_per_block):
            if not loss_mask.bool().any():
                break
            talk_outputs = self.model(
                input_ids=None,
                inputs_embeds=talk_input_embeds,
                inputs_repres=talk_rps,
                attention_mask=talk_attn_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            nfe += 1
            logits = talk_outputs.logits.float()
            talk_rps = talk_outputs.hidden_states

            if self.soft_inputs_enabled:
                talk_input_ids, talk_input_embeds, loss_mask = denoise_k_step_soft_embed_v2(
                    input_ids=talk_input_ids,
                    target=None,
                    loss_mask=loss_mask,
                    logits=logits,
                    emb_weight=self.model.talk_embed_weight,
                    k_reveal=self.reveal_k,
                    soft_topk=self.soft_topk,
                    soft_temp=self.soft_temp,
                    mode=self.reveal_cfg.get("policy", "ar_force"),
                    decode_cfg=self.decode_cfg,
                )
            else:
                talk_input_ids, loss_mask, _, _ = denoise_k_step_hard(
                    input_ids=talk_input_ids,
                    target=None,
                    loss_mask=loss_mask,
                    logits=logits,
                    reveal_cfg=self.reveal_cfg,
                    decode_cfg=self.decode_cfg,
                )
                talk_input_embeds = F.embedding(talk_input_ids, self.model.talk_embed_weight)

        return talk_input_ids, nfe


def _build_blockwise_attention_bias(
    *,
    seq_len: int,
    max_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Block-causal allow mask: prompt is bidirectional within itself; each
    generated block sees prompt + all earlier blocks + itself; future mask
    blocks are blocked. Returns `[1, 1, max_len, max_len]` bool. Caller
    slices to the relevant prefix per block forward.
    """
    bias = torch.zeros((max_len, max_len), dtype=torch.bool, device=device)
    bias[:seq_len, :seq_len] = True
    num_blocks = (max_len - seq_len) // block_size
    for b in range(num_blocks):
        end = seq_len + (b + 1) * block_size
        bias[seq_len + b * block_size : end, :end] = True
    return bias.unsqueeze(0).unsqueeze(0)


def _trim_kv_cache_trailing(past_key_values, n: int):
    """Drop the last `n` time-steps from each layer's K and V.

    The KV cache returned from the think pass includes K/V for the trailing
    mask block. Those entries become stale once we decode that block — the
    next think pass needs to recompute them against the freshly revealed
    tokens — so we trim them here.
    """
    return [
        tuple(kv[:, :, :-n] for kv in layer)
        for layer in past_key_values
    ]


def _select_talk_inputs(
    x0: torch.Tensor,
    think_rps: torch.Tensor,
    *,
    block_size: int,
    block_start: int,
    block_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the per-block input_ids and hidden representation feeding the talk model.

    For block 0 the prompt is shorter than the (block_start, block_end) span,
    so the slice falls back to the trailing `block_size` positions of the
    think output. For all later blocks, the (block_start, block_end) slice
    points into the full-sequence x0/think_rps.
    """
    if x0.shape[-1] > 2 * block_size:
        return x0[:, block_start:block_end], think_rps[:, block_start:block_end, :]
    return x0[:, -block_size:], think_rps[:, -block_size:, :]

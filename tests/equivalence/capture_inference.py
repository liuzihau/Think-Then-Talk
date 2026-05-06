"""Capture deterministic inference output for axis 3 equivalence gating.

Runs T3 inference on a fixed prompt + seed and writes the generated token IDs
to a JSON file. Designed to be executable on **both pre-axis-3 and post-axis-3
commits** so the same script can produce baseline and refactor captures.

Two decode paths:

    --mode inline   Reproduces the pre-axis-3 inference loop directly against
                    `T3Model` (dense block-causal attention_bias, no engine).
                    Available on every commit that has T3Model — i.e. before
                    and after the axis 3 refactor.

    --mode engine   Uses `model.inference_engine.T3DecodeEngine`. Requires
                    axis 3 (commit 4e7dfc5 or later) and exercises the
                    `prefer_no_mask` fast path on commit B (eb4ad38) onward.

Usage:

    python tests/equivalence/capture_inference.py \
        --ckpt_path /path/to/state_2 \
        --prompt "Solve: 17 + 35 = ?" \
        --gen_length 48 --block_size 6 --seed 0 \
        --mode inline \
        --output /tmp/baseline.json

The output JSON is:

    {
      "ckpt_path": str, "prompt": str, "seed": int,
      "gen_length": int, "block_size": int, "mode": "inline" | "engine",
      "prompt_token_ids": [int, ...],
      "generated_token_ids": [int, ...],
      "decoded_text": str,
      "nfe": int,
      "git_commit": str,
    }

Generated tokens and decoded text are saved together because either alone is
ambiguous: tokens are exact but unhumane; text is friendly but hides
tokenizer-edge ambiguities.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Imports that exist in BOTH pre- and post-axis-3 codebases.
from model.modeling_t3 import T3Model
from utils import (
    denoise_k_step_hard,
    denoise_k_step_soft_embed_v2,
    get_denoise_decode_config,
    get_denoise_reveal_config,
    load_ckpt,
)


LLADA_MASK_TOKEN_ID = 126336


def _set_seed(seed: int) -> None:
    """Inline copy: utils.set_seed only exists post-axis-3."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "<unavailable>"


# --- Inline decode (works on pre- and post-axis-3) ----------------------------

def _build_blockwise_attention_bias(
    seq_len: int, max_len: int, block_size: int, device: torch.device,
) -> torch.Tensor:
    """Block-causal allow mask: [1,1,max_len,max_len] bool. Same shape the
    pre-axis-3 inference.py and eval_t3.py built inline.
    """
    bias = torch.zeros((max_len, max_len), dtype=torch.bool, device=device)
    bias[:seq_len, :seq_len] = True
    num_blocks = (max_len - seq_len) // block_size
    for b in range(num_blocks):
        end = seq_len + (b + 1) * block_size
        bias[seq_len + b * block_size : end, :end] = True
    return bias.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def _decode_inline(
    model: T3Model, model_config: dict, prompt_ids: torch.Tensor,
    *, gen_length: int, block_size: int,
) -> tuple[torch.Tensor, int]:
    """Reimplements the pre-axis-3 inference.py per-block decode loop directly.

    Independent of T3DecodeEngine, so it produces identical output before and
    after axis 3 (assuming the model itself is unchanged).
    """
    device = prompt_ids.device
    seq_len = prompt_ids.shape[1]
    max_len = seq_len + gen_length

    denoise_cfg = model_config.get("denoise", {}) or {}
    reveal_cfg = get_denoise_reveal_config(denoise_cfg)
    decode_cfg = get_denoise_decode_config(denoise_cfg, default_k=reveal_cfg["k"])
    reveal_k = int(reveal_cfg["k"])
    soft_cfg = model_config.get("soft_inputs", {}) or {}
    soft_inputs_enabled = bool(soft_cfg.get("enabled", False))

    x = torch.full((1, max_len), LLADA_MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :seq_len] = prompt_ids
    position_ids = torch.arange(0, max_len, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, max_len), dtype=torch.bool, device=device)
    attention_bias = _build_blockwise_attention_bias(
        seq_len=seq_len, max_len=max_len, block_size=block_size, device=device,
    )

    past_key_values = None
    cur_len = seq_len + block_size
    x0 = x[:, :cur_len]
    p0 = position_ids[:, :cur_len]

    nfe = 0
    num_blocks = gen_length // block_size
    for block_idx in range(num_blocks):
        think_outputs = model(
            input_ids=x0,
            attention_mask=attention_mask[:, :cur_len],
            attention_bias=attention_bias[:, :, :cur_len, :cur_len],
            position_ids=p0,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        nfe += 1

        # Trim trailing block from KV cache.
        past_key_values = [
            tuple(kv[:, :, :-block_size] for kv in layer)
            for layer in think_outputs.past_key_values
        ]

        block_start = seq_len + block_size * block_idx
        block_end = block_start + block_size
        if x0.shape[-1] > 2 * block_size:
            talk_input_ids = x0[:, block_start:block_end]
            talk_rps = think_outputs.hidden_states[:, block_start:block_end, :]
        else:
            talk_input_ids = x0[:, -block_size:]
            talk_rps = think_outputs.hidden_states[:, -block_size:, :]

        loss_mask = torch.ones_like(talk_input_ids, dtype=torch.float32)
        talk_attn_mask = torch.ones_like(talk_input_ids, dtype=torch.long)
        talk_input_embeds = F.embedding(talk_input_ids, model.talk_embed_weight)

        for _ in range(int(denoise_cfg.get("steps", model.length))):
            if not loss_mask.bool().any():
                break
            talk_outputs = model(
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

            if soft_inputs_enabled:
                talk_input_ids, talk_input_embeds, loss_mask = denoise_k_step_soft_embed_v2(
                    input_ids=talk_input_ids, target=None, loss_mask=loss_mask, logits=logits,
                    emb_weight=model.talk_embed_weight, k_reveal=reveal_k,
                    soft_topk=int(soft_cfg.get("top_k", 0)),
                    soft_temp=float(soft_cfg.get("temperature", 1.0)),
                    mode=reveal_cfg.get("policy", "ar_force"),
                    decode_cfg=decode_cfg,
                )
            else:
                talk_input_ids, loss_mask, _, _ = denoise_k_step_hard(
                    input_ids=talk_input_ids, target=None, loss_mask=loss_mask, logits=logits,
                    reveal_cfg=reveal_cfg, decode_cfg=decode_cfg,
                )
                talk_input_embeds = F.embedding(talk_input_ids, model.talk_embed_weight)

        x[:, block_start:block_end] = talk_input_ids
        cur_len = block_end + block_size
        x0 = x[:, block_start:cur_len]
        p0 = position_ids[:, block_start:cur_len]

    return x, nfe


# --- Engine decode (post-axis-3 only) -----------------------------------------

@torch.no_grad()
def _decode_engine(
    model: T3Model, model_config: dict, prompt_ids: torch.Tensor,
    *, gen_length: int, block_size: int,
) -> tuple[torch.Tensor, int]:
    from model.inference_engine import T3DecodeEngine  # post-axis-3 only
    engine = T3DecodeEngine(
        model=model, model_config=model_config,
        block_size=block_size, gen_length=gen_length,
    )
    return engine.generate(prompt_ids)


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--prompt", default="Solve: 17 + 35 = ?")
    p.add_argument("--gen_length", type=int, default=48)
    p.add_argument("--block_size", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=["inline", "engine"], default="inline")
    p.add_argument("--output", required=True, help="Path to write JSON capture")
    args = p.parse_args()

    _set_seed(args.seed)

    with (Path(args.ckpt_path) / "config.json").open() as f:
        model_config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["pretrained_model_name_or_path"], trust_remote_code=True,
    )
    msgs = [{"role": "user", "content": args.prompt}]
    chat = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(
        chat, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(args.device)
    prompt_len = int(prompt_ids.shape[1])

    model = T3Model(model_config, train=False, device=args.device)
    model.eval()
    load_ckpt(args.ckpt_path, model, optimizer=None, scheduler=None, map_location="cpu")

    decode = _decode_engine if args.mode == "engine" else _decode_inline
    t0 = time.time()
    x, nfe = decode(
        model, model_config, prompt_ids,
        gen_length=args.gen_length, block_size=args.block_size,
    )
    elapsed = time.time() - t0

    gen_ids = x[0, prompt_len:].tolist()
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    payload = {
        "ckpt_path": args.ckpt_path,
        "prompt": args.prompt,
        "seed": args.seed,
        "gen_length": args.gen_length,
        "block_size": args.block_size,
        "mode": args.mode,
        "prompt_token_ids": prompt_ids[0].tolist(),
        "generated_token_ids": gen_ids,
        "decoded_text": decoded,
        "nfe": int(nfe),
        "elapsed_seconds": round(elapsed, 3),
        "git_commit": _git_commit(),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[capture] mode={args.mode}  nfe={nfe}  elapsed={elapsed:.2f}s")
    print(f"[capture] generated tokens ({len(gen_ids)}): {gen_ids[:16]}{' ...' if len(gen_ids) > 16 else ''}")
    print(f"[capture] decoded text: {decoded[:120]}{' ...' if len(decoded) > 120 else ''}")
    print(f"[capture] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())

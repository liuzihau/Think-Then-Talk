"""End-to-end inference smoke for axis 3.

Sanity-checks that `T3DecodeEngine.generate` runs on a real checkpoint and
produces the expected output shape with the prompt segment intact and the
generated segment non-trivial. This is *not* an equivalence test — that lives
in `tests/equivalence/`.

Run from the repo root:

    T3_CKPT_PATH=/path/to/state_2 pytest tests/test_inference_smoke.py -v

Optional env:
    T3_SMOKE_PROMPT       Prompt text. Default: "Solve: 17 + 35 = ?"
    T3_SMOKE_GEN_LENGTH   Tokens to generate. Default: 24 (4 blocks at bs=6).
    T3_SMOKE_BLOCK_SIZE   Default 6.
    T3_SMOKE_SEED         Default 0.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from model.inference_engine import T3DecodeEngine, LLADA_MASK_TOKEN_ID
from model.modeling_t3_infer import T3InferenceModel
from utils import load_ckpt, set_seed


def _ckpt_dir() -> Path:
    raw = os.environ.get("T3_CKPT_PATH", "").strip()
    if not raw:
        pytest.skip("Set T3_CKPT_PATH=/path/to/state_N")
    p = Path(raw.split(":", 1)[0])
    if not p.is_dir():
        pytest.fail(f"{p} is not a directory")
    return p


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for T3 inference smoke")


@pytest.fixture(scope="module")
def loaded_engine():
    _need_cuda()
    ckpt = _ckpt_dir()
    with (ckpt / "config.json").open() as f:
        config = json.load(f)

    set_seed(int(os.environ.get("T3_SMOKE_SEED", "0")))
    model = T3InferenceModel(config, train=False, device="cuda:0")
    model.eval()
    load_ckpt(str(ckpt), model, optimizer=None, scheduler=None, map_location="cpu")

    engine = T3DecodeEngine(
        model=model,
        model_config=config,
        block_size=int(os.environ.get("T3_SMOKE_BLOCK_SIZE", "6")),
        gen_length=int(os.environ.get("T3_SMOKE_GEN_LENGTH", "24")),
    )
    return model, config, engine


def _build_prompt_ids(model_config: dict, device: str) -> torch.Tensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_config["pretrained_model_name_or_path"], trust_remote_code=True,
    )
    text = os.environ.get("T3_SMOKE_PROMPT", "Solve: 17 + 35 = ?")
    msgs = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def test_engine_produces_expected_shape(loaded_engine):
    """`generate` returns `[1, prompt_len + gen_length]` and reports nfe > 0."""
    _, config, engine = loaded_engine
    prompt_ids = _build_prompt_ids(config, device="cuda:0")
    prompt_len = prompt_ids.shape[1]

    out, nfe = engine.generate(prompt_ids)

    assert out.shape == (1, prompt_len + engine.gen_length), out.shape
    assert nfe >= engine.num_blocks, (
        f"nfe={nfe} < num_blocks={engine.num_blocks} — think pass missed?"
    )


def test_prompt_segment_preserved(loaded_engine):
    """Engine must not overwrite the prompt segment of `x`."""
    _, config, engine = loaded_engine
    prompt_ids = _build_prompt_ids(config, device="cuda:0")
    prompt_len = prompt_ids.shape[1]

    out, _ = engine.generate(prompt_ids)
    assert torch.equal(out[:, :prompt_len], prompt_ids), (
        "Prompt segment was modified by generate(); engine should only write to [prompt_len:]"
    )


def test_generation_segment_non_trivial(loaded_engine):
    """At least one generated token differs from the mask id (decode actually ran)."""
    _, config, engine = loaded_engine
    prompt_ids = _build_prompt_ids(config, device="cuda:0")
    prompt_len = prompt_ids.shape[1]

    out, _ = engine.generate(prompt_ids)
    gen = out[0, prompt_len:]
    assert (gen != LLADA_MASK_TOKEN_ID).any(), (
        "All generated tokens are mask id — denoise loop did not reveal anything"
    )

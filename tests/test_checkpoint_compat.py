"""Checkpoint key-drift gate for axis 3.

Loads an existing T3 checkpoint through both `T3Model` and `T3InferenceModel`
and asserts that no `nn.Module` attribute was renamed during the refactor —
i.e. the talk-side `state_dict()` keys match exactly between the saved tensor
and the freshly built model.

Run from the repo root:

    T3_CKPT_PATH=/path/to/state_2 pytest tests/test_checkpoint_compat.py -v

Multiple checkpoints (separator: `:`) are also supported:

    T3_CKPT_PATH=/p/state_0:/p/state_2 pytest tests/test_checkpoint_compat.py -v

Talk-model keys allowed to be missing in older checkpoints (kept current init):
    rps_eta_param  — added when scalable-eta param was introduced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from model.modeling_t3 import T3InferenceModel, T3Model


_ALLOW_MISSING_TALK = {"rps_eta_param"}


def _ckpt_paths() -> list[Path]:
    raw = os.environ.get("T3_CKPT_PATH", "").strip()
    if not raw:
        return []
    return [Path(p) for p in raw.split(":") if p]


def _need_ckpt():
    if not _ckpt_paths():
        pytest.skip("Set T3_CKPT_PATH=/path/to/state_N (colon-separated for many)")


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (T3Model materialises bf16 weights on GPU)")


def _load_config(ckpt_dir: Path) -> dict:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        pytest.fail(f"{cfg_path} missing — checkpoint dir is incomplete")
    with cfg_path.open() as f:
        return json.load(f)


def _talk_state_keys(ckpt_dir: Path) -> set[str]:
    payload = torch.load(ckpt_dir / "ckpt.pt", map_location="cpu")
    if "talk_model" not in payload:
        pytest.fail(f"{ckpt_dir}/ckpt.pt has no 'talk_model' key")
    return set(payload["talk_model"].keys())


@pytest.fixture(params=_ckpt_paths(), ids=lambda p: p.name)
def ckpt_dir(request) -> Path:
    _need_ckpt()
    p = request.param
    if not p.is_dir():
        pytest.fail(f"{p} is not a directory")
    return p


@pytest.mark.parametrize("model_cls", [T3Model, T3InferenceModel],
                         ids=["T3Model", "T3InferenceModel"])
def test_talk_state_dict_keys_match(ckpt_dir: Path, model_cls):
    """Saved talk_model keys exactly match the freshly built model's keys.

    Allows `rps_eta_param` to be missing from the checkpoint (added later;
    `load_ckpt` keeps the current init for it). Anything else missing or
    unexpected is a key-drift bug introduced by the refactor.
    """
    _need_cuda()
    config = _load_config(ckpt_dir)
    saved_keys = _talk_state_keys(ckpt_dir)

    if model_cls is T3InferenceModel:
        model = model_cls(config, train=False, device="cuda:0")
    else:
        model = model_cls(config, train=False, device="cuda:0")

    model_keys = set(model.talk_model.state_dict().keys())
    missing = sorted((model_keys - saved_keys) - _ALLOW_MISSING_TALK)
    unexpected = sorted(saved_keys - model_keys)

    assert not missing, (
        f"talk_model keys present in fresh {model_cls.__name__} but absent in "
        f"checkpoint at {ckpt_dir}: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
    )
    assert not unexpected, (
        f"talk_model keys present in checkpoint at {ckpt_dir} but absent in "
        f"fresh {model_cls.__name__}: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
    )

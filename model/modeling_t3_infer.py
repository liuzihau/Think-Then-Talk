"""Inference-focused T3 runtime.

Thin subclass of `T3Model` that disables activation checkpointing and resolves
`device="auto"` against accelerate / `LOCAL_RANK`. Checkpoint-compatible with
the training-side `T3Model` — no `nn.Module` attributes are added or renamed,
so existing `state_dict()`s load with zero key drift.
"""
import os

import torch

from model.modeling_t3 import T3Model


def _read_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def resolve_inference_device(device: str | torch.device) -> str | torch.device:
    """Resolve `"auto"` against the runtime environment.

    - `"auto"` + no CUDA visible → `"cpu"`
    - `"auto"` + multi-GPU launch (WORLD_SIZE > 1, LOCAL_RANK set) → `"cuda:{LOCAL_RANK}"`
    - `"auto"` + single GPU → `"cuda:0"`
    - any other value: returned unchanged.
    """
    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = _read_int_env("LOCAL_RANK")
    world_size = _read_int_env("WORLD_SIZE") or 1
    if world_size > 1 and local_rank is not None:
        return f"cuda:{local_rank}"
    return "cuda:0"


class T3InferenceModel(T3Model):
    """T3Model with activation checkpointing disabled. Identical state_dict layout."""

    def __init__(
        self,
        config: dict,
        dtype: torch.dtype = torch.bfloat16,
        train: bool = False,
        device: str | torch.device = "auto",
    ):
        if isinstance(device, str):
            device = resolve_inference_device(device)
        super().__init__(config=config, dtype=dtype, train=train, device=device)
        if self.architecture == "LLaDA":
            self.think_model_root.model.set_activation_checkpointing(None)

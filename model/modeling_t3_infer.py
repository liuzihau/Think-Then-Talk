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


def resolve_inference_device(device: str) -> str:
    """Resolve a single inference device from an explicit string or "auto".

    "auto" picks `cuda:LOCAL_RANK` under torchrun/accelerate, `cuda:0` for a
    single visible GPU, and `cpu` when no GPU is available.
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
    """
    Inference-focused T3 runtime.

    Goals:
    - keep checkpoint compatibility with the training-time T3 model
    - disable activation checkpointing during eval/inference
    - resolve a device from accelerate/env when the caller leaves it on "auto"
    """

    def __init__(
        self,
        config: dict,
        dtype: torch.dtype = torch.bfloat16,
        train: bool = False,
        device: str | torch.device = "auto",
    ):
        if isinstance(device, str):
            device = resolve_inference_device(device)
        print(f"[T3InferenceModel] resolved device: {device}")

        super().__init__(
            config=config,
            dtype=dtype,
            train=train,
            device=device,
        )

        if self.architecture == "LLaDA":
            self.think_model_root.model.set_activation_checkpointing(None)

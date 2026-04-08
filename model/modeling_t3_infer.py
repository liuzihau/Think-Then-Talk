import os
from typing import Tuple

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


def _count_visible_cuda_devices() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def resolve_inference_devices(
    think_dev1: str,
    think_dev2: str,
    talk_dev: str,
) -> Tuple[str, str, str]:
    explicit = [think_dev1, think_dev2, talk_dev]
    if all(dev != "auto" for dev in explicit):
        return think_dev1, think_dev2, talk_dev

    num_devices = _count_visible_cuda_devices()
    if num_devices <= 0:
        return "cpu", "cpu", "cpu"

    local_rank = _read_int_env("LOCAL_RANK")
    world_size = _read_int_env("WORLD_SIZE") or 1

    if world_size > 1 and local_rank is not None:
        local_device = f"cuda:{local_rank}"
        resolved = [
            local_device if dev == "auto" else dev
            for dev in explicit
        ]
        return tuple(resolved)  # type: ignore[return-value]

    if num_devices == 1:
        resolved = [
            "cuda:0" if dev == "auto" else dev
            for dev in explicit
        ]
        return tuple(resolved)  # type: ignore[return-value]

    default_triplet = ("cuda:0", "cuda:1", "cuda:1")
    resolved = [
        default_triplet[idx] if dev == "auto" else dev
        for idx, dev in enumerate(explicit)
    ]
    return tuple(resolved)  # type: ignore[return-value]


class T3InferenceModel(T3Model):
    """
    Inference-focused T3 runtime.

    Goals:
    - keep checkpoint compatibility with the training-time T3 model
    - disable activation checkpointing during eval/inference
    - resolve devices from accelerate/env when the caller leaves them on "auto"
    """

    def __init__(
        self,
        config: dict,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict = "auto",
        train: bool = False,
        think_dev1: str = "auto",
        think_dev2: str = "auto",
        talk_dev: str = "auto",
    ):
        think_dev1, think_dev2, talk_dev = resolve_inference_devices(
            think_dev1=think_dev1,
            think_dev2=think_dev2,
            talk_dev=talk_dev,
        )
        print(
            "[T3InferenceModel] resolved devices: "
            f"think_device1={think_dev1}, think_device2={think_dev2}, talk_device={talk_dev}"
        )

        super().__init__(
            config=config,
            dtype=dtype,
            device_map=device_map,
            train=train,
            think_dev1=think_dev1,
            think_dev2=think_dev2,
            talk_dev=talk_dev,
        )

        if self.architecture == "LLaDA":
            self.think_model_root.model.set_activation_checkpointing(None)

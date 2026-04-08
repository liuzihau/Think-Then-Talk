import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, Sampler

from train.data_process import build_dataset_rank


NEMOTRON_REPO = "nvidia/Llama-Nemotron-Post-Training-Dataset"
BLOCK_NAMES = ("nemo_chat", "nemo_code", "nemo_math", "small_mixed")


@dataclass(frozen=True)
class SourceSpec:
    datapath: str
    split: str
    train_subset_percent: Optional[str]
    test_subset_percent: Optional[str]
    train_subset_offset: Optional[str]
    test_subset_offset: Optional[str]
    assistant_think_drop_prob: Optional[str]


def _parse_csv(v: Optional[str], n: int, name: str) -> List[Optional[str]]:
    if v is None:
        return [None] * n
    parts = [x.strip() for x in str(v).split(",")]
    if len(parts) == 1 and n > 1:
        parts = parts * n
    if len(parts) != n:
        raise ValueError(
            f"{name} has {len(parts)} values, but train_dataset has {n}. "
            "Provide one value or one per original dataset entry."
        )
    return [p if p != "" else None for p in parts]


def _parse_source_specs(data_cfg: Dict[str, Any]) -> List[SourceSpec]:
    paths = [p.strip() for p in str(data_cfg["train_dataset"]).split(",") if p.strip()]
    splits = [p.strip() for p in str(data_cfg.get("splits", "chat")).split(",") if p.strip()]
    if len(splits) == 1 and len(paths) > 1:
        splits = splits * len(paths)
    if len(paths) != len(splits):
        raise ValueError(
            f"train_dataset has {len(paths)} entries but splits has {len(splits)} entries."
        )

    train_pcts = _parse_csv(data_cfg.get("train_subset_percents"), len(paths), "train_subset_percents")
    test_pcts = _parse_csv(data_cfg.get("test_subset_percents"), len(paths), "test_subset_percents")
    train_offsets = _parse_csv(data_cfg.get("train_subset_offsets"), len(paths), "train_subset_offsets")
    test_offsets = _parse_csv(data_cfg.get("test_subset_offsets"), len(paths), "test_subset_offsets")
    think_drop_probs = _parse_csv(
        data_cfg.get("assistant_think_drop_probs"),
        len(paths),
        "assistant_think_drop_probs",
    )

    return [
        SourceSpec(
            datapath=paths[i],
            split=splits[i],
            train_subset_percent=train_pcts[i],
            test_subset_percent=test_pcts[i],
            train_subset_offset=train_offsets[i],
            test_subset_offset=test_offsets[i],
            assistant_think_drop_prob=think_drop_probs[i],
        )
        for i in range(len(paths))
    ]


def _repo_id(datapath: str) -> str:
    if ":" in datapath:
        return datapath.split(":", 1)[0].strip()
    return datapath.strip()


def _block_index_for_spec(spec: SourceSpec) -> int:
    repo_id = _repo_id(spec.datapath)
    if repo_id == NEMOTRON_REPO and spec.split == "chat":
        return 0
    if repo_id == NEMOTRON_REPO and spec.split == "code":
        return 1
    if repo_id == NEMOTRON_REPO and spec.split == "math":
        return 2
    return 3


def _build_single_source_dataset(
    tokenizer,
    spec: SourceSpec,
    *,
    max_len: int,
    target_len: int,
    data_cfg: Dict[str, Any],
    seed: int,
    get_test_subset: bool,
):
    kwargs = dict(
        splits=spec.split,
        seed=seed,
        test_split_ratio=float(data_cfg.get("test_split_ratio", 0.05)),
        short_target_mode=data_cfg.get("short_target_mode", "skip"),
        strip_nemotron_math_prompt_prefix=data_cfg.get("strip_nemotron_math_prompt_prefix", False),
        train_subset_percents=spec.train_subset_percent,
        test_subset_percents=spec.test_subset_percent,
        train_subset_offsets=spec.train_subset_offset,
        test_subset_offsets=spec.test_subset_offset,
        assistant_think_drop_probs=spec.assistant_think_drop_prob,
    )
    return build_dataset_rank(
        tokenizer,
        spec.datapath,
        max_len,
        target_len,
        get_test_subset=get_test_subset,
        **kwargs,
    )


def _concat_or_raise(datasets: Sequence, block_name: str):
    if not datasets:
        raise ValueError(f"Balanced 4-block sampler requires non-empty block '{block_name}'.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(list(datasets))


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx):
        raise IndexError("EmptyDataset has no items")


class FourBlockBalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        *,
        block_lengths: Sequence[int],
        batch_size: int,
        seed: int = 42,
        anchor_block: int = 3,
    ):
        if len(block_lengths) != 4:
            raise ValueError(f"Expected 4 block lengths, got {len(block_lengths)}.")
        if batch_size % 4 != 0:
            raise ValueError(f"Balanced 4-block sampling requires batch_size % 4 == 0, got {batch_size}.")
        if any(length <= 0 for length in block_lengths):
            raise ValueError(f"Each block must be non-empty, got lengths={list(block_lengths)}.")

        self.block_lengths = [int(x) for x in block_lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.anchor_block = int(anchor_block)
        self.samples_per_block = self.batch_size // 4
        self.block_offsets = []
        offset = 0
        for length in self.block_lengths:
            self.block_offsets.append(offset)
            offset += length
        self.num_batches = max(1, math.ceil(self.block_lengths[self.anchor_block] / self.samples_per_block))
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        orders = [torch.randperm(length, generator=generator).tolist() for length in self.block_lengths]
        cursors = [0] * 4

        def draw_from_block(block_idx: int) -> List[int]:
            picked: List[int] = []
            while len(picked) < self.samples_per_block:
                order = orders[block_idx]
                cursor = cursors[block_idx]
                remaining = len(order) - cursor
                need = self.samples_per_block - len(picked)
                if remaining == 0:
                    orders[block_idx] = torch.randperm(
                        self.block_lengths[block_idx],
                        generator=generator,
                    ).tolist()
                    cursors[block_idx] = 0
                    continue
                take = min(remaining, need)
                local_ids = order[cursor:cursor + take]
                cursors[block_idx] += take
                picked.extend(self.block_offsets[block_idx] + local_id for local_id in local_ids)
            return picked

        for _ in range(self.num_batches):
            batch: List[int] = []
            for block_idx in range(4):
                batch.extend(draw_from_block(block_idx))
            yield batch


def build_4block_balanced_train_and_test_datasets(
    tokenizer,
    data_cfg: Dict[str, Any],
    *,
    max_len: int,
    target_len: int,
    seed: int,
    batch_size: int,
):
    source_specs = _parse_source_specs(data_cfg)
    grouped_specs: List[List[SourceSpec]] = [[] for _ in range(4)]
    for spec in source_specs:
        grouped_specs[_block_index_for_spec(spec)].append(spec)

    train_block_datasets = []
    test_parts_all = []
    block_lengths: List[int] = []

    for block_idx, specs in enumerate(grouped_specs):
        train_parts = [
            _build_single_source_dataset(
                tokenizer,
                spec,
                max_len=max_len,
                target_len=target_len,
                data_cfg=data_cfg,
                seed=seed,
                get_test_subset=False,
            )
            for spec in specs
        ]
        test_parts = [
            _build_single_source_dataset(
                tokenizer,
                spec,
                max_len=max_len,
                target_len=target_len,
                data_cfg=data_cfg,
                seed=seed,
                get_test_subset=True,
            )
            for spec in specs
        ]
        train_block = _concat_or_raise(train_parts, BLOCK_NAMES[block_idx])
        train_block_datasets.append(train_block)
        block_lengths.append(len(train_block))
        test_parts_all.extend([part for part in test_parts if len(part) > 0])

    train_dataset = ConcatDataset(train_block_datasets)
    if not test_parts_all:
        test_dataset = EmptyDataset()
    elif len(test_parts_all) == 1:
        test_dataset = test_parts_all[0]
    else:
        test_dataset = ConcatDataset(test_parts_all)
    batch_sampler = FourBlockBalancedBatchSampler(
        block_lengths=block_lengths,
        batch_size=batch_size,
        seed=seed,
        anchor_block=3,
    )
    block_info = {
        BLOCK_NAMES[i]: block_lengths[i]
        for i in range(4)
    }
    return train_dataset, batch_sampler, test_dataset, block_info

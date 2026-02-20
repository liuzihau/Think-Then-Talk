import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
import matplotlib.pyplot as plt


def build_dataset_rank(
    tokenizer,
    datapaths: str,
    max_len: int,
    target_len: int,
    *,
    splits: str = "chat",
    cache_root: str = "./hf_datasets_cache",
    num_proc: int = 8,
    test_split_ratio: float = 0.05,
    get_test_subset: bool = False,
    seed: int = 42,
):
    """
    datapaths:
      - comma-separated list of:
        - local path to a dataset saved with save_to_disk(), OR
        - HF Hub dataset name (optionally with config, e.g. "org/name" or "org/name:config")

    Output format (unchanged):
      - HuggingFace Dataset with columns: ["input_ids", "target", "attention_mask"]
      - set_format(type="torch") is applied before returning

    Caching:
      1) raw dataset cached to disk under cache_root/<repo_or_path>/<split>
      2) processed dataset cached to disk under cache_root/processed/<cache_key>/(train|test)
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something "
        "not correct. If you don't know the answer to a question, please don't share false information."
    )

    SEP = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ROLES_OK = {"human", "user", "gpt", "assistant"}

    def _is_local_saved_dataset(p: str) -> bool:
        path = Path(p)
        if not path.exists() or not path.is_dir():
            return False
        return (
            (path / "dataset_info.json").exists()
            or (path / "state.json").exists()
            or (path / "data").exists()
        )

    def _parse_hf_id(s: str) -> Tuple[str, Optional[str]]:
        s = s.strip()
        if ":" in s:
            repo_id, config = s.split(":", 1)
            repo_id, config = repo_id.strip(), config.strip()
            return repo_id, (config if config else None)
        return s, None

    def _safe_dirname(s: str) -> str:
        s = s.strip()
        return re.sub(r"[^\w\-.]+", "_", s)

    def _ensure_pad_token():
        # keep your behaviour: if pad token missing, use unk
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id

    def _apply_chat_template(messages: List[Dict[str, str]]) -> str:
        # keep your behaviour: apply_chat_template + removesuffix(assistant header)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).removesuffix("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def _tokenize_ids(text: str) -> List[int]:
        _ensure_pad_token()
        ids = tokenizer(
            text,
            return_tensors=None,          # return python lists
            add_special_tokens=False,
        )["input_ids"]
        # tokenizer can return list[int] for single string
        return ids

    def _build_prompt_and_target(conversation: str) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """
        Convert a full conversation string (ending with an assistant answer)
        into:
          input_ids: prompt ids (up to assistant header)
          target: assistant answer token ids
          attention_mask: ones for prompt length
        Returns None if filtered out.
        """
        full_ids = _tokenize_ids(conversation)

        if len(full_ids) > max_len:
            return None

        turns = conversation.split(SEP)
        if len(turns) < 2:
            return None

        prompt = ""
        for turn in turns[:-1]:
            prompt += turn + SEP

        input_ids = _tokenize_ids(prompt)
        target = full_ids[len(input_ids):]

        if len(target) < target_len:
            return None

        attention_mask = [1] * len(input_ids)
        return input_ids, target, attention_mask

    # -----------------------------
    # Dataset-specific “adapters”
    # -----------------------------
    def _iter_conversations_from_batch(datapath: str, examples: Dict) -> List[str]:
        """
        Produce a list of conversation strings to be turned into (prompt, target).
        Each string is a full conversation that includes the assistant answer at the end.
        """
        conversations: List[str] = []

        if datapath == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            data_pts = len(examples.get("input", []))
            for i in range(data_pts):
                source = examples["input"][i]
                response = examples["output"][i]

                if not source or source[0].get("role") not in ROLES_OK:
                    # keep your debug print behaviour, but avoid crashing
                    try:
                        print(source[0].get("role"))
                    except Exception:
                        print("bad_source_role")
                    continue

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in source:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "assistant", "content": response})

                conversations.append(_apply_chat_template(messages))

            return conversations

        if datapath == "allenai/tulu-3-sft-mixture":
            data_pts = len(examples.get("messages", []))
            for i in range(data_pts):
                source = examples["messages"][i]
                if not source:
                    continue

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if source[0].get("role") == "system":
                    messages[0] = source[0]
                    source = source[1:]

                # Create a training instance at every assistant turn (same as your original)
                for msg in source:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    if msg.get("role") == "assistant":
                        conversations.append(_apply_chat_template(messages))

            return conversations

        raise ValueError(f"Unsupported datapath for preprocessing: {datapath}")

    def _make_preprocess_fn(datapath: str) -> Callable[[Dict], Dict]:
        """
        Returns a batched map function producing exactly:
          input_ids, target, attention_mask (lists, not tensors)
        """
        def _fn(examples: Dict) -> Dict:
            new_examples = {"attention_mask": [], "target": [], "input_ids": []}

            conversations = _iter_conversations_from_batch(datapath, examples)
            for conv in conversations:
                out = _build_prompt_and_target(conv)
                if out is None:
                    continue
                input_ids, target, attention_mask = out
                new_examples["input_ids"].append(input_ids)
                new_examples["target"].append(target)
                new_examples["attention_mask"].append(attention_mask)

            return new_examples

        return _fn

    # -----------------------------
    # Resolve datapaths
    # -----------------------------
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    paths = [p.strip() for p in datapaths.split(",") if p.strip()]
    all_splits = [p.strip() for p in splits.split(",") if p.strip()]

    if not paths:
        raise ValueError("datapaths is empty. Provide at least one dataset path or HF dataset id.")

    final_ds = None

    for datapath, split in zip(paths, all_splits):
        # 1) Load raw dataset (from disk or HF and cache)
        if _is_local_saved_dataset(datapath):
            ds = load_from_disk(datapath)
        else:
            repo_id, config = _parse_hf_id(datapath)
            cache_key = repo_id if config is None else f"{repo_id}:{config}"
            raw_local_path = cache_root / _safe_dirname(cache_key) / split

            if _is_local_saved_dataset(str(raw_local_path)):
                ds = load_from_disk(str(raw_local_path))
            else:
                if config is None:
                    ds_hf = load_dataset(repo_id, split=split)
                else:
                    ds_hf = load_dataset(repo_id, config, split=split)

                raw_local_path.mkdir(parents=True, exist_ok=True)
                ds_hf.save_to_disk(str(raw_local_path))
                ds = load_from_disk(str(raw_local_path))

        # 2) Shuffle then split (same logic)
        ds = ds.shuffle(seed=seed)

        if test_split_ratio > 0 and len(ds) > 1:
            splits = ds.train_test_split(test_size=test_split_ratio, seed=seed)
            ds1 = splits["test"] if get_test_subset else splits["train"]
            print(
                f"dataset rank: returning {'TEST' if get_test_subset else 'TRAIN'} split ({len(ds1)} examples)"
            )
        else:
            ds1 = ds
            print(f"dataset rank: returning FULL dataset ({len(ds1)} examples)")

        original_columns = ds1.column_names

        # 3) Processed caching (IMPORTANT: include tokenizer identity)
        processed_root = cache_root / "processed"
        processed_root.mkdir(parents=True, exist_ok=True)

        tok_id = getattr(tokenizer, "name_or_path", "unknown_tokenizer")
        proc_key = _safe_dirname(f"{datapath}:{split}:{tok_id}:max{max_len}:tgt{target_len}:seed{seed}:v1")
        processed_path = processed_root / proc_key / ("test" if get_test_subset else "train")
        print(processed_path)

        if _is_local_saved_dataset(str(processed_path)):
            ds1_proc = load_from_disk(str(processed_path))
        else:
            preprocess_fn = _make_preprocess_fn(datapath)

            ds1_proc = ds1.map(
                preprocess_fn,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns,
                load_from_cache_file=False,  # keep your behaviour: rely on save_to_disk instead
            )
            ds1_proc.save_to_disk(str(processed_path))

        # 4) Output format unchanged
        ds1_proc.set_format(type="torch")

        # 5) Concat across datapaths
        if final_ds is None:
            final_ds = ds1_proc
        else:
            final_ds = concatenate_datasets([final_ds, ds1_proc])

    return final_ds


class DataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]], length: int = 4, mask_token_id: int = 126336, pad_token_id: int = 126081) -> Dict[str, Any]:
        # ---- helpers: accept [T] or [1,T] and always use [T]
        def _to_1d(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2 and x.size(0) == 1:
                return x.squeeze(0)
            return x

        B = len(features)
        device = features[0]["input_ids"].device

        input_ids_list = [_to_1d(f["input_ids"]) for f in features]          # [Li]
        attn_mask_list = [_to_1d(f["attention_mask"]) for f in features]     # [Li]
        target_list    = [_to_1d(f["target"]) for f in features]             # [Ti]

        # ---- sample start indices uniformly for each example
        # valid starts: 0..Ti-length (inclusive) => count = Ti-length+1
        max_starts = torch.tensor(
            [max(t.size(0) - length + 1, 1) for t in target_list],
            device=device
        )  # [B], clamp to >=1 to avoid errors if Ti < length

        # uniform integer in [0, max_starts[i)-1]
        starts = (torch.rand(B, device=device) * max_starts).long()  # [B]

        # # TODO turn off debug
        # starts = torch.ones((B,)).long() * 5

        # ---- compute final sequence lengths and max_length
        seq_lens = []
        for i in range(B):
            prefix_len = int(starts[i].item())
            seq_lens.append(input_ids_list[i].size(0) + prefix_len + length)
        max_length = max(seq_lens)

        # ---- allocate batch tensors (more efficient than per-item pad+cat)
        dtype_ids = input_ids_list[0].dtype  # usually torch.long
        batch_input_ids = torch.full((B, max_length), pad_token_id, dtype=dtype_ids, device=device)
        batch_attention_mask = torch.zeros((B, max_length), dtype=attn_mask_list[0].dtype, device=device)
        batch_loss_mask = torch.zeros((B, max_length), dtype=torch.long, device=device)  # usually bool/long

        # target window: [B, length]
        batch_target = torch.empty((B, length), dtype=target_list[0].dtype, device=device)

        mask_tokens = torch.full((length,), mask_token_id, dtype=dtype_ids, device=device)

        # ---- fill each row
        for i in range(B):
            inp = input_ids_list[i]
            att = attn_mask_list[i]
            tgt = target_list[i]

            s = int(starts[i].item())
            # clamp in case tgt shorter than length
            s = min(s, max(tgt.size(0) - length, 0))

            prefix = tgt[:s]  # [s]
            window = tgt[s:s + length]  # [length] (or shorter if tgt too short)

            # if tgt is shorter than length, pad window (rare if your data is valid)
            if window.size(0) < length:
                padded = torch.full((length,), pad_token_id, dtype=tgt.dtype, device=device)
                padded[:window.size(0)] = window
                window = padded

            seq = torch.cat([inp, prefix.to(dtype_ids), mask_tokens], dim=0)  # [seq_len]
            L = seq.size(0)

            batch_input_ids[i, :L] = seq
            batch_attention_mask[i, :L] = 1  # or: torch.cat([att, ones...]) if you need original attn pattern
            batch_loss_mask[i, L - length:L] = 1  # only mask tokens contribute to loss
            batch_target[i] = window

        return {
            "input_ids": batch_input_ids,
            "target": batch_target,                 # [B, length]
            "attention_mask": batch_attention_mask, # [B, max_length]
            "loss_mask": batch_loss_mask,           # [B, max_length]
        }


def build_block_attention_mask(
    max_length: int,
    inp_len: int,
    prefix_len: int,
    window_len: int,
    mask_len: int,
    block_size: int,
    device=None,
    allow_mask_within_block: bool = True,
    inp_attend_everything: bool = True,
) -> torch.BoolTensor:
    """
    Returns attn_mask of shape [S, S] where True means 'can attend'.
    Sequence layout:
      [ inp | prefix | window | mask ]
    with prefix/window blockwise-causal, mask blockwise growth as described.
    """
    device = device or "cpu"

    # Total length
    S = inp_len + prefix_len + window_len + mask_len
    m = torch.zeros((max_length, max_length), dtype=torch.bool, device=device)

    # Offsets
    o_inp = 0
    o_pre = o_inp + inp_len
    o_win = o_pre + prefix_len
    o_msk = o_win + window_len

    # Helper: allow rows [r0:r1) to attend to cols [c0:c1)
    def allow(r0, r1, c0, c1, v=True):
        if r1 > r0 and c1 > c0:
            m[r0:r1, c0:c1] = v

    # -----------------------
    # 1) inp region
    # -----------------------
    # Everyone can attend to inp (inp is always visible context)
    r0 = o_inp
    r1 = o_inp + inp_len
    c0 = o_inp
    c1 = o_inp + inp_len
    allow(r0, r1, c0, c1)#, 0.2)

    # -----------------------
    # 2) prefix region: blockwise-causal
    # -----------------------
    # Blocks are contiguous chunks of size block_size.
    assert prefix_len % block_size == 0
    num_pre_blocks = (prefix_len) // block_size
    for b in range(num_pre_blocks):
        r0 = o_pre + b * block_size
        r1 = o_pre + (b + 1) * block_size
        c0 = o_inp
        c1 = o_pre + (b + 1) * block_size
        allow(r0, r1, c0, c1)#, 0.5)

    # -----------------------
    # 3) window region: same blockwise-causal as prefix
    # -----------------------
    assert window_len % block_size == 0
    num_win_blocks = window_len // block_size
    for b in range(num_win_blocks):
        r0 = o_win + b * block_size
        r1 = o_win + (b + 1) * block_size
        c0 = o_inp
        c1 = o_win + (b + 1) * block_size
        allow(r0, r1, c0, c1)#, 0.7)

    # -----------------------
    # 4) mask region: growth by window blocks
    # -----------------------
    assert mask_len % block_size == 0
    num_mask_blocks = mask_len // block_size
    for b in range(num_mask_blocks):
        r0 = o_msk + b * block_size
        r1 = o_msk + (b + 1) * block_size
        c0 = o_inp
        c1 = o_win + b * block_size
        allow(r0, r1, c0, c1)#, 1)

        c0 = o_msk + b * block_size
        c1 = o_msk + (b + 1) * block_size
        allow(r0, r1, c0, c1)#, 1)


    return m

class DataCollatorWithPaddingV2:
    def __init__(self, block_size: int=32, block_num: int=8, mask_token_id: int=126336, pad_token_id: int=126081, eos_token_id: int=126081, start_end_ratio: float=0.2, max_start_mode: str="exceed_end"):
        self.block_size = block_size
        self.block_num = block_num
        self.total_length = self.block_size * self.block_num
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        self.neg_inf = torch.finfo(torch.float32).min
        self.start_end_ratio = start_end_ratio
        self.max_start_mode = max_start_mode

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ---- helpers: accept [T] or [1,T] and always use [T]
        def _to_1d(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2 and x.size(0) == 1:
                return x.squeeze(0)
            return x

        B = len(features)
        device = features[0]["input_ids"].device

        input_ids_list = [_to_1d(f["input_ids"]) for f in features]          # [Li]
        attn_mask_list = [_to_1d(f["attention_mask"]) for f in features]     # [Li]
        target_list    = [_to_1d(f["target"]) for f in features]             # [Ti]

        # for b in range(B):
        #     print(input_ids_list[b].shape)
        #     print(attn_mask_list[b].shape)
        #     print(target_list[b].shape)
        # raise Exception("hahaha")
        
        # ---- sample start indices uniformly for each example
        # valid starts: 0..Ti-length (inclusive) => count = Ti-length+1
        normal_max_starts = torch.tensor(
                [max(t.size(0) - self.block_size * self.block_num + 1, 1) for t in target_list],
                device=device
        )  # [B], clamp to >=1 to avoid errors if Ti < length
        end_max_starts = torch.tensor(
                [max(t.size(0) - (self.block_size * self.block_num * 3) // 4 + 1, 1)for t in target_list],
                device=device
        )

        # uniform integer in [0, max_starts[i)-1]
        roll = torch.rand(B, device=device)
        p = self.start_end_ratio / 2.0

        starts = torch.zeros(B, device=device, dtype=torch.long)

        mask_start = roll < p
        mask_end   = roll > (1.0 - p)
        mask_rand  = ~(mask_start | mask_end)

        starts[mask_start] = 0
        starts[mask_end]   = end_max_starts[mask_end] - 1

        # independent uniform draw for random region
        u = torch.rand(mask_rand.sum(), device=device)
        starts[mask_rand] = (u * normal_max_starts[mask_rand].float()).long()

        for b in range(B):
            starts[b] =  (starts[b] // self.block_size) * self.block_size
        
        # ---- compute final sequence lengths and max_length
        # prefix + generated target + [block_size * block_num] gt + [block_size * block_num] mask
        seq_lens = []
        for i in range(B):
            prefix_len = int(starts[i].item())
            seq_lens.append(input_ids_list[i].size(0) + prefix_len + 2 * self.total_length - self.block_size)
        max_length = max(seq_lens)

        # ---- allocate batch tensors (more efficient than per-item pad+cat)
        dtype_ids = input_ids_list[0].dtype  # usually torch.long
        batch_input_ids = torch.full((B, max_length), self.pad_token_id, dtype=dtype_ids, device=device)
        batch_position_ids = torch.zeros((B, max_length), dtype=dtype_ids, device=device)
        batch_attention_mask = torch.zeros((B, max_length), dtype=torch.bool, device=device)
        batch_attention_bias = torch.empty((B, 1, max_length, max_length), dtype=torch.float32, device=device)
        batch_loss_mask = torch.zeros((B, max_length), dtype=torch.long, device=device)  # usually bool/long

        # target window: [B, length]
        batch_target = torch.empty((B, self.total_length), dtype=target_list[0].dtype, device=device)

        mask_tokens = torch.full((self.total_length,), self.mask_token_id, dtype=dtype_ids, device=device)

        # ---- fill each row
        for i in range(B):
            inp = input_ids_list[i]
            att = attn_mask_list[i]
            tgt = target_list[i]

            s = int(starts[i].item())
            # clamp in case tgt shorter than length
            # s = (min(s, max(tgt.size(0) - self.total_length, 0)) // self.block_size) * self.block_size

            prefix = tgt[:s]  # [s]
            window = tgt[s:s + self.total_length]  # [length] (or shorter if tgt too short)
                        
            # if tgt is shorter than length, pad window (rare if your data is valid)
            # window_size = window.size(0)
            # if window_size < self.total_length:
            #     window_size = (window.size(0) // self.block_size) * self.block_size
            #     if window_size == 0:
            #         raise ValueError("a target window with size 0")
            #     window = window[:window_size]

            real_len = window.size(0)
            if real_len < self.total_length:
                eos_pad = torch.full((self.total_length - real_len,), self.eos_token_id, dtype=tgt.dtype, device=device)
                window = torch.cat([window, eos_pad], dim=0)
            window_size = self.total_length
            
            seq = torch.cat([inp, prefix.to(dtype_ids), window[:-self.block_size], mask_tokens[:window_size]], dim=0)  # [seq_len]
            L = seq.size(0)

            prefix_end = inp.shape[0] + prefix.shape[0]
            prefix_pos = torch.arange(0, prefix_end)
            window_pos = torch.arange(prefix_end, prefix_end + window_size-self.block_size)
            masked_pos = torch.arange(prefix_end, prefix_end + window_size)
            pos = torch.cat([prefix_pos, window_pos, masked_pos], dim=0)  # [seq_len]
            assert pos.shape[0] == L, "mismatch length in position_ids and input_ids"
            
            batch_input_ids[i, :L] = seq
            batch_position_ids[i, :L] = pos
            
            batch_attention_mask[i, :L] = 1
            allow = build_block_attention_mask(
                max_length=max_length,
                inp_len=inp.shape[0],
                prefix_len=s,
                window_len=window[:-self.block_size].shape[0],
                mask_len=mask_tokens[:window_size].shape[0],
                block_size=self.block_size,
                device=inp.device,
                allow_mask_within_block=True,
                inp_attend_everything=False,
            )
            bias = torch.where(
                allow,
                torch.zeros_like(allow, dtype=torch.float32),
                torch.full_like(allow, self.neg_inf, dtype=torch.float32),
            )
            batch_attention_bias[i, 0, :allow.shape[0], :allow.shape[1]] = bias
            batch_loss_mask[i, L - window_size:L] = 1  # only mask tokens contribute to loss
            batch_target[i] = window


            # save_attn_mask_fig(batch_attention_mask, f"debug/attn_mask_{i}.png", title="Block attention check")

        return {
            "input_ids": batch_input_ids,
            "position_ids": batch_position_ids,
            "target": batch_target,                 # [B, length]
            "attention_mask": batch_attention_mask, # [B, max_length]
            "attention_bias": batch_attention_bias, # [B, :, max_length, max_length]
            "loss_mask": batch_loss_mask,           # [B, max_length]
        }


def save_attn_mask_fig(
    attn_mask: torch.Tensor,
    save_path: str,
    title: str = "Attention mask (1=allow, 0=block)",
    max_side: int = 2048,
    show: bool = False,
    dpi: int = 200,
):
    """
    Save attention mask visualization to file.

    Supports:
      - bool [S,S] where True = allow
      - bool [B,S,S] (uses batch 0)
      - additive float/int [S,S] where 0=allow and -inf/very negative=block
      - additive float/int [B,S,S] (uses batch 0)

    Args:
        save_path: path to save image, e.g. "mask.png"
        show: whether to also display the figure
    """
    save_path = Path(save_path)

    # pick batch 0 if batched
    if attn_mask.dim() == 3:
        attn = attn_mask[0]
    else:
        attn = attn_mask

    if attn.shape[-2] != attn.shape[-1]:
        raise ValueError(f"Expected square mask [S,S], got {tuple(attn.shape)}")

    S = attn.shape[-1]

    # downsample if too large
    if S > max_side:
        step = int(np.ceil(S / max_side))
        attn = attn[::step, ::step]
        title = f"{title} (downsampled {step}x, original S={S})"

    # Convert to 1=allow, 0=block
    if attn.dtype == torch.bool:
        img = attn.detach().cpu().numpy().astype(np.float32)
    else:
        img = attn.detach().cpu().numpy()

    plt.figure(figsize=(7, 7))
    plt.imshow(img, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.xlabel("Key index (attend-to)")
    plt.ylabel("Query index (attend-from)")
    plt.colorbar(label="1=allow, 0=block")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    testdataset = build_dataset_rank(
        tokenizer, "nvidia/Llama-Nemotron-Post-Training-Dataset,allenai/tulu-3-sft-mixture", 4096, 256, splits="chat,train",
        get_test_subset=True, seed=42
    )
    test_loader = DataLoader(
        testdataset,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        collate_fn=DataCollatorWithPaddingV2()
    )
    print(len(test_loader))
    for data in test_loader:
        for key in data:
            print(key, data[key].dtype)
        break
    
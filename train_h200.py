# -*- coding: utf-8 -*-
"""
Single-device T3 training, wrapped with 🤗 Accelerate.
Single-GPU is plain `accelerator.prepare()`; 2-GPU is the same code launched
with `--num_processes=2` (DDP). No DeepSpeed, no ZeRO, no FSDP.
"""

import os, re, json, inspect, subprocess, sys
import argparse
import math
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from model.modeling_t3 import T3Model
from model.attention.flex_block_mask import build_t3_block_mask
from train.data_process import build_dataset_rank, DataCollatorWithPaddingV2
from train.data_process_balanced import build_4block_balanced_train_and_test_datasets
from train.visualize import visualize_t3_batch_trace
from utils import (
    AttrDict,
    denoise_k_step_hard,
    denoise_k_step_soft_embed_v2,
    get_denoise_decode_config,
    get_denoise_reveal_config,
    get_policy_label,
    load_ckpt,
    save_ckpt,
    topk_soft_embedding_from_logits,
)


# -----------------------------
# Utilities
# -----------------------------
def freeze_parameters(model: torch.nn.Module):
    # ---- Think model: LoRA only ----
    for n, p in model.think_model.named_parameters():
        if "lora_" in n.lower():
            p.requires_grad = True
            print("[THINK][LoRA]", n)
        else:
            p.requires_grad = False

    # ---- Talk model: all trainable ----
    for n, p in model.talk_model.named_parameters():
        p.requires_grad = True
        print("[TALK]", n)

    # ---- Stats ----
    def count_params(m):
        tot = sum(p.numel() for p in m.parameters())
        tr  = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return tot, tr

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== PARAM SUMMARY ===")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"think_model  total/trainable: {count_params(model.think_model)}")
    print(f"talk_model   total/trainable: {count_params(model.talk_model)}")


def calculate_ploss(out_logp, target, loss_mask):
    """
    out_logp: [B, L, V]
    target : [B, L]
    loss_mask: [B, L] (0/1 or float/bool)
    """
    target_indices = target.unsqueeze(-1).long()  # [B, L, 1]
    target_logp = out_logp.gather(2, target_indices).squeeze(2)  # [B, L]
    mask = loss_mask.float()
    return -(target_logp * mask).sum() / (mask.sum().clamp_min(1e-6))

def ploss_sum_and_count(out_logp, target, loss_mask):
    # out_logp: [BG, L, V], target: [BG, L], loss_mask: [BG, L] (1=active)
    target_logp = out_logp.gather(2, target.unsqueeze(-1).long()).squeeze(-1)  # [BG, L]
    m = loss_mask.float()
    loss_sum = -(target_logp * m).sum()
    count = m.sum()
    return loss_sum, count

def build_step_weights(loss_cfg: dict, num_steps: int, epoch: int):
    # base weights
    step_agg = loss_cfg.get("step_agg", {})
    if not step_agg.get("enabled", True):
        w = torch.ones(num_steps, dtype=torch.float32)
    else:
        weight_type = step_agg.get("weight_type", "exp_decay")
        cap_step = int(step_agg.get("cap_step", num_steps - 1))
        min_w = float(step_agg.get("min_weight", 0.0))
        max_w = float(step_agg.get("max_weight", 1e9))

        if weight_type == "exp_decay":
            base = float(step_agg.get("base", 1.0))
            w = []
            for i in range(num_steps):
                eff_i = min(i, cap_step)
                w.append(base ** eff_i)
            w = torch.tensor(w, dtype=torch.float32)
        elif weight_type == "manual":
            manual_w = step_agg.get("manual_weights", [])
            if not isinstance(manual_w, list) or len(manual_w) == 0:
                raise ValueError("step_agg.manual_weights must be a non-empty list when weight_type='manual'")
            if len(manual_w) < num_steps:
                raise ValueError(
                    f"manual_weights length ({len(manual_w)}) must be at least num_steps ({num_steps})"
                )
            # Training can finish in fewer denoise iterations than the configured maximum
            # once every target position has been revealed, so we only use the executed prefix.
            w = torch.tensor([float(x) for x in manual_w[:num_steps]], dtype=torch.float32)
        elif weight_type == "uniform":
            w = torch.ones(num_steps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        w = torch.clamp(w, min=min_w, max=max_w)

    # step0 boost
    s0 = loss_cfg.get("step0_boost", {})
    if s0.get("enabled", False) and num_steps > 0:
        mode = s0.get("mode", "linear_smooth")
        start_mul = float(s0.get("start_multiplier", 1.0))
        end_mul = float(s0.get("end_multiplier", 1.0))
        over_epochs = int(s0.get("over_epochs", 1))

        if mode == "linear_smooth":
            # epoch 0 -> start_mul, epoch >= over_epochs -> end_mul
            t = min(max(epoch, 0), over_epochs)
            alpha = t / max(1, over_epochs)
            mul = (1 - alpha) * start_mul + alpha * end_mul
        elif mode == "constant":
            mul = start_mul
        else:
            raise ValueError(f"Unknown step0_boost mode: {mode}")

        w[0] *= mul

    return w  # torch.float32 [num_steps]

def per_step_loss_sum_and_count(loss_cfg: dict, logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor):
    """
    logits: [BG, L, V] float
    target: [BG, L] long
    loss_mask: [BG, L] float/bool (1=active)
    returns: (loss_sum, count) both scalars tensors
    """
    loss_type = loss_cfg.get("type", "token_nll")

    if loss_type == "token_nll":
        out_logp = F.log_softmax(logits, dim=-1)  # [BG, L, V]
        target_logp = out_logp.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [BG, L]
        m = loss_mask.float()
        loss_sum = -(target_logp * m).sum()
        count = m.sum()

    elif loss_type == "ce":
        # standard CE, with mask applied manually
        # NOTE: you can add label_smoothing support if needed
        V = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [BG, L]
        m = loss_mask.float()
        loss_sum = (nll * m).sum()
        count = m.sum()

    else:
        raise ValueError(f"Unknown loss.type: {loss_type}")

    return loss_sum, count

def calculate_correct_counts(logits, target, loss_mask):
    pred = logits.argmax(dim=-1)
    correct = (pred == target) & loss_mask.bool()
    return correct.sum().item(), loss_mask.sum().item()

def pick_single_position(loss_mask: torch.Tensor, logits: torch.Tensor | None, mode: str):
    """
    Pick exactly one active position per row.
    Returns:
      cols: [BG] selected column index per row
      valid: [BG] bool, whether that row has at least one active position
    """
    device = loss_mask.device
    BG, L = loss_mask.shape
    active = loss_mask.bool()

    if mode == "ar_force":
        pos = torch.arange(L, device=device).unsqueeze(0).expand(BG, L)
        scores = (-pos).to(torch.float32)
    elif mode == "random":
        scores = torch.rand((BG, L), device=device)
    elif mode == "greedy":
        if logits is None:
            raise ValueError("mode='greedy' requires logits")
        scores = logits.max(dim=-1).values
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scores = scores.masked_fill(~active, float("-inf"))
    cols = scores.argmax(dim=1)
    valid = active.any(dim=1)
    return cols, valid

def selected_position_loss_acc(
    logits: torch.Tensor,      # [BG, L, V]
    target: torch.Tensor,      # [BG, L]
    cols: torch.Tensor,        # [BG]
    valid: torch.Tensor,       # [BG] bool
):
    """
    Compute token NLL and accuracy at selected positions only.
    """
    if not valid.any():
        return 0.0, 0.0

    BG = logits.size(0)
    rows = torch.arange(BG, device=logits.device)

    sel_target = target[rows, cols].long()
    sel_pred = logits.argmax(dim=-1)[rows, cols]
    sel_logp = F.log_softmax(logits, dim=-1)[rows, cols, sel_target]

    sel_acc = (sel_pred == sel_target).float()[valid]
    sel_loss = (-sel_logp)[valid]

    return float(sel_loss.mean().item()), float(sel_acc.mean().item())

def masked_position_loss_acc(
    logits: torch.Tensor,         # [BG, L, V]
    target: torch.Tensor,         # [BG, L]
    position_mask: torch.Tensor,  # [BG, L]
):
    """
    Compute token NLL and accuracy over an arbitrary subset of positions.
    """
    m = position_mask.bool()
    if not m.any():
        return 0.0, 0.0

    target_logp = F.log_softmax(logits, dim=-1).gather(2, target.unsqueeze(-1)).squeeze(-1)
    pred = logits.argmax(dim=-1)
    loss = (-(target_logp[m])).mean().item()
    acc = (pred[m] == target[m]).float().mean().item()
    return float(loss), float(acc)

def append_jsonl(path: str, payload: dict):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_eval_model_args(ckpt_path: str, block_size: int, device: str, eval_cfg: dict):
    """Build the JSON `--model_args` payload consumed by eval_t3.py."""
    device_str = str(eval_cfg.get("device", device))
    model_args = {
        "ckpt_path": ckpt_path,
        "gen_length": int(eval_cfg.get("gen_length", 512)),
        "block_size": int(eval_cfg.get("block_size", block_size)),
        "device": device_str,
        "show_speed": bool(eval_cfg.get("show_speed", True)),
        "prompt_prefix": str(eval_cfg.get("prompt_prefix", "")),
        "prompt_suffix": str(eval_cfg.get("prompt_suffix", "")),
    }
    return json.dumps(model_args)


def run_inference_suite(train_config: dict, savedir: str, state_dir: str, epoch: int, device: str):
    inference_cfg = train_config.get("inference", {})
    if not inference_cfg.get("enabled", False):
        return

    every_n_epochs = int(inference_cfg.get("every_n_epochs", 1))
    if every_n_epochs > 1 and ((epoch + 1) % every_n_epochs != 0):
        print(f"[Inference] skip at epoch {epoch + 1}: every_n_epochs={every_n_epochs}")
        return

    evals = inference_cfg.get("evals", [])
    if not isinstance(evals, list) or len(evals) == 0:
        print("[Inference] enabled but no evals configured; skipping.")
        return

    script_dir = Path(__file__).resolve().parent
    eval_script = script_dir / "eval_t3.py"
    output_root = inference_cfg.get("output_root", "evals_results")
    log_path = os.path.join(savedir, "logs", "inference_runs.jsonl")

    for eval_cfg in evals:
        if not eval_cfg.get("enabled", True):
            continue

        task = str(eval_cfg["task"])
        num_fewshot = int(eval_cfg.get("num_fewshot", 0))
        gen_length = int(eval_cfg.get("gen_length", 512))
        batch_size = int(eval_cfg.get("batch_size", 1))
        limit = eval_cfg.get("limit", None)
        output_path = eval_cfg.get(
            "output_path",
            os.path.join(savedir, output_root, f"{task}-ns{num_fewshot}-gl{gen_length}"),
        )

        cmd = [
            sys.executable,
            str(eval_script),
            "--model",
            "t3_model",
            "--model_args",
            build_eval_model_args(
                ckpt_path=state_dir,
                block_size=train_config["data"]["block_size"],
                device=device,
                eval_cfg=eval_cfg,
            ),
            "--tasks",
            task,
            "--num_fewshot",
            str(num_fewshot),
            "--batch_size",
            str(batch_size),
            "--output_path",
            output_path,
        ]
        if limit is not None and str(limit) != "":
            cmd.extend(["--limit", str(limit)])
        if eval_cfg.get("log_samples", False):
            cmd.append("--log_samples")
        if eval_cfg.get("confirm_run_unsafe_code", False):
            cmd.append("--confirm_run_unsafe_code")

        print(f"[Inference] epoch={epoch + 1} task={task} limit={limit} output_path={output_path}")
        print("[Inference] cmd:", " ".join(cmd))
        status = "ok"
        error_msg = None
        try:
            subprocess.run(cmd, cwd=str(script_dir), check=True)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error_msg = str(exc)
            print(f"[Inference] task={task} failed: {exc}")
            if inference_cfg.get("fail_on_error", False):
                raise

        append_jsonl(
            log_path,
            {
                "epoch": int(epoch),
                "state_dir": state_dir,
                "task": task,
                "output_path": output_path,
                "limit": limit,
                "status": status,
                "error": error_msg,
            },
        )


def find_max_state_with_file(directory, filename="ckpt.pt"):
    """
    Look for: <savedir>/state_<N>/<filename>
    Return (path_to_state_dir, next_epoch_start)
    """
    if not os.path.isdir(directory):
        return None, 0

    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)

    if max_a == -1:
        return None, 0
    return os.path.join(directory, f"state_{max_a}"), max_a + 1

def resolve_state_dir(path: str | None, filename: str = "ckpt.pt"):
    """
    Resolve a checkpoint directory from user-provided path.
    Accepts:
      - a state dir that directly contains <filename>
      - a run root directory that contains state_<N>/<filename> (picks max N)
      - a direct path to <filename>
    Returns state_dir or None.
    """
    if path is None:
        return None
    if not isinstance(path, str):
        return None
    path = os.path.expandvars(os.path.expanduser(path.strip()))
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if os.path.isfile(path):
        if os.path.basename(path) == filename:
            return os.path.dirname(path)
        return None

    if os.path.isdir(path):
        direct_file = os.path.join(path, filename)
        if os.path.isfile(direct_file):
            return path
        found_state_dir, _ = find_max_state_with_file(path, filename=filename)
        return found_state_dir

    return None


def denoise_k_step(
    input_ids,
    target,
    loss_mask,
    logits=None,
    k=1,
    mode="random",
    decode_cfg=None,
    generator=None,
):
    """
    input_ids: [BG, L]
    target:    [BG, L]
    loss_mask: [BG, L] (0/1 float or bool)
    """
    reveal_cfg = {"k": int(k), "policy": mode}
    input_ids, loss_mask, _, _ = denoise_k_step_hard(
        input_ids=input_ids,
        target=target,
        loss_mask=loss_mask,
        logits=logits,
        reveal_cfg=reveal_cfg,
        decode_cfg=decode_cfg or {"policy": "fix", "fix_k": int(k)},
        generator=generator,
    )
    return input_ids, loss_mask


def denoise_k_step_soft_embed(
    input_ids: torch.Tensor,        # [BG, L]
    target: torch.Tensor,           # [BG, L]
    loss_mask: torch.Tensor,        # [BG, L] (0/1 float/bool) 1=still masked
    logits: torch.Tensor,           # [BG, L, V] from current step
    emb_weight: torch.Tensor,       # [V, D] (your talk_embed_weight)
    k_reveal: int = 1,
    topk: int = 32,
    temperature: float = 1.0,
    generator=None,
):
    """
    Returns:
      input_ids_next        [BG, L]   (for logging / optional use)
      input_emb_next        [BG, L, D] (feed this to talk model next step)
      loss_mask_next        [BG, L]
    """
    device = input_ids.device
    BG, L = input_ids.shape

    # --- pick k positions to teacher-force reveal ---
    active = loss_mask.bool()
    if generator is not None:
        assert generator.device == device
        scores = torch.rand((BG, L), device=device, generator=generator)
    else:
        scores = torch.rand((BG, L), device=device)
    scores = scores.masked_fill(~active, float("-inf"))

    idx = scores.topk(k=min(k_reveal, L), dim=1).indices  # [BG, k]
    chosen_active = active.gather(1, idx)                  # [BG, k]

    rows = torch.arange(BG, device=device).unsqueeze(1).expand_as(idx)
    rows = rows[chosen_active]
    cols = idx[chosen_active]

    # --- update ids/mask (bookkeeping) ---
    input_ids_next = input_ids.clone()
    loss_mask_next = loss_mask.clone()

    input_ids_next[rows, cols] = target[rows, cols]
    loss_mask_next[rows, cols] = 0

    # --- build next-step embeddings ---
    # Base: embed current tokens (after reveal)
    base_emb = F.embedding(input_ids_next, emb_weight).to(dtype=emb_weight.dtype)  # [BG, L, D]

    # Soft emb from logits for "still masked" positions
    soft_emb = topk_soft_embedding_from_logits(
        logits=logits,
        emb_weight=emb_weight,
        topk=topk,
        temperature=temperature,
    ).to(dtype=emb_weight.dtype)                              # [BG, L, D]

    # For revealed positions, use GT embedding; for still-masked, use soft embedding
    m = loss_mask_next.bool().unsqueeze(-1)                   # [BG, L, 1]
    input_emb_next = torch.where(m, soft_emb, base_emb)       # [BG, L, D]

    return input_ids_next, input_emb_next, loss_mask_next

def assert_module_on_device(module: torch.nn.Module, device: str, name: str):
    dev = next(module.parameters()).device
    if str(dev) != device:
        raise RuntimeError(f"{name} is on {dev}, expected {device}")

def detach_state(rps, input_embeds):
    if rps is not None:
        rps = rps.detach()
    if input_embeds is not None:
        input_embeds = input_embeds.detach()
    return rps, input_embeds

def grad_norms_of_lora(model, k=10):
    items = []
    for n, p in model.named_parameters():
        if "lora_" in n and p.requires_grad:
            g = None if p.grad is None else p.grad.detach().float().norm().item()
            items.append((n, g))
    items = sorted(items, key=lambda x: (x[1] is None, -(x[1] or 0.0)))
    for n, g in items[:k]:
        print(f"{n:80s} grad_norm={g}")
    none_cnt = sum(g is None for _, g in items)
    print("lora grads None:", none_cnt, "/", len(items))

# -----------------------------
# FlexAttention BlockMask cache
# -----------------------------
# Building a BlockMask is fast (~ms), but with `train_micro_batch_size_per_gpu=1`
# many batches share the exact same per-sample region offsets, so caching by
# shape signature gives a near-100% hit rate.

_BLOCK_MASK_CACHE: dict = {}
_BLOCK_MASK_CACHE_LIMIT = 256


def get_block_mask_for_batch(data: dict, device):
    """Build (or fetch from cache) the FlexAttention BlockMask for this batch."""
    key = (
        int(data["max_length"]),
        int(data["block_size"]),
        tuple(int(x) for x in data["inp_len"].tolist()),
        tuple(int(x) for x in data["prefix_len"].tolist()),
        tuple(int(x) for x in data["window_len"].tolist()),
        tuple(int(x) for x in data["mask_len"].tolist()),
    )
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is None:
        bm = build_t3_block_mask(
            inp_len=data["inp_len"],
            prefix_len=data["prefix_len"],
            window_len=data["window_len"],
            mask_len=data["mask_len"],
            block_size=int(data["block_size"]),
            seq_len=int(data["max_length"]),
            device=device,
        )
        if len(_BLOCK_MASK_CACHE) >= _BLOCK_MASK_CACHE_LIMIT:
            _BLOCK_MASK_CACHE.clear()
        _BLOCK_MASK_CACHE[key] = bm
    return bm


def split_params(model):
    talk_params = []
    lora_params = []
    lm_head_params = []
    other_trainable = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        if n in ("talk_lm_head_weight", "talk_lm_head_bias"):
            lm_head_params.append(p)
        elif "lora" in nl:
            lora_params.append(p)
        elif "talk_model" in nl:
            talk_params.append(p)
        else:
            other_trainable.append(p)

    return talk_params, lora_params, lm_head_params, other_trainable

# -----------------------------
# Main
# -----------------------------
def main():
    SEED = 0

    parser = argparse.ArgumentParser()
    # parser.add_argument("--trainpath", type=str, default="nvidia/Llama-Nemotron-Post-Training-Dataset,allenai/tulu-3-sft-mixture")
    # parser.add_argument("--testpath", type=str, default="nvidia/Llama-Nemotron-Post-Training-Dataset,allenai/tulu-3-sft-mixture")
    # parser.add_argument("--trainpath", type=str, default="nvidia/Llama-Nemotron-Post-Training-Dataset")
    # parser.add_argument("--testpath", type=str, default="nvidia/Llama-Nemotron-Post-Training-Dataset")
    parser.add_argument("--split", type=str, default="chat,train")
    parser.add_argument("--savedir", type=str, default="0")
    parser.add_argument("--training_config", type=str, default="train/train_config.json")
    parser.add_argument("--model_config", type=str, default="model/config.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    VIS_EVERY_OPT_STEP = 2000     # optimiser steps
    VIS_MAX_STEPS = 16            # denoise steps to print (avoid huge logs)
    VIS_BIDX = 0                  # which sample in batch
    STEP_METRIC_LOG = "logs/train_step_metrics.jsonl"
    EPOCH_METRIC_LOG = "logs/epoch_metrics.jsonl"

    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(SEED)

    # -------------------------
    # Config
    # -------------------------
    with open(args.training_config) as f:
        train_config = json.load(f)
    training_parameters = AttrDict(train_config["training_parameters"])
    with open(args.model_config) as f:
        model_config = json.load(f)

    # copy model related parameter from train_config
    model_config["train_dataset"] = train_config["data"]["train_dataset"]
    model_config["length"] = train_config["data"]["block_size"]
    model_config["prune_last_n_layer"] = train_config["talk_model"]["prune_last_n_layer"]
    model_config["talk_model"]["n_layers"] = train_config["talk_model"]["n_layers"]
    model_config["mix_indexes"] = train_config["mix_indexes"]
    model_config["denoise"] = train_config["denoise"]
    if "train_lm_head" in train_config:
        model_config["train_lm_head"] = train_config["train_lm_head"]
    if train_config["lora"]["enabled"]:
        model_config['lora'] = train_config["lora"]
    if train_config['rps_residual']['enabled']:
        model_config['rps_residual'] = train_config['rps_residual']
    if train_config['soft_inputs']['enabled']:
        model_config['soft_inputs'] = train_config['soft_inputs']

    # -------------------------
    # Accelerator
    # -------------------------
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
    )
    device = accelerator.device

    # -------------------------
    # Model
    # -------------------------
    if args.savedir == "0":
        args.savedir = f'{training_parameters["wandb_name"]}-{train_config["data"]["block_size"]}-{train_config["data"]["block_num"]}-{train_config["gradient_accumulation_steps"]}'
    os.makedirs(args.savedir, exist_ok=True)

    model = T3Model(model_config, device=device)

    freeze_parameters(model)

    # Eval/train modes
    model.train()
    model.think_model.train()
    model.talk_model.train()

    # If no ckpt, init talking
    ckpt_dir, start_epoch = find_max_state_with_file(args.savedir, filename="ckpt.pt")
    init_from_state_dir_cfg = train_config.get("init_from_state_dir", None)

    # -------------------------
    # Tokenizer / Data
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_model_name_or_path"], trust_remote_code=True)
    tokenizer.eos_token_id = 126348
    data_cfg = train_config["data"]
    dataset_splits = data_cfg.get("splits", args.split)
    short_target_mode = data_cfg.get("short_target_mode", "skip")
    build_dataset_rank_params = inspect.signature(build_dataset_rank).parameters
    dataset_common_kwargs = dict(
        splits=dataset_splits,
        seed=SEED,
    )
    if "test_split_ratio" in build_dataset_rank_params:
        dataset_common_kwargs["test_split_ratio"] = data_cfg.get("test_split_ratio", 0.05)
    if "train_subset_percents" in build_dataset_rank_params:
        dataset_common_kwargs["train_subset_percents"] = data_cfg.get("train_subset_percents", None)
    if "test_subset_percents" in build_dataset_rank_params:
        dataset_common_kwargs["test_subset_percents"] = data_cfg.get("test_subset_percents", None)
    if "train_subset_offsets" in build_dataset_rank_params:
        dataset_common_kwargs["train_subset_offsets"] = data_cfg.get("train_subset_offsets", None)
    if "test_subset_offsets" in build_dataset_rank_params:
        dataset_common_kwargs["test_subset_offsets"] = data_cfg.get("test_subset_offsets", None)
    if "short_target_mode" in build_dataset_rank_params:
        dataset_common_kwargs["short_target_mode"] = short_target_mode
    else:
        print("[Data] build_dataset_rank has no short_target_mode; using legacy behavior.")
    if "strip_nemotron_math_prompt_prefix" in build_dataset_rank_params:
        dataset_common_kwargs["strip_nemotron_math_prompt_prefix"] = data_cfg.get("strip_nemotron_math_prompt_prefix", False)
    if "assistant_think_drop_probs" in build_dataset_rank_params:
        dataset_common_kwargs["assistant_think_drop_probs"] = data_cfg.get("assistant_think_drop_probs", None)

    sampling_mode = data_cfg.get("sampling_mode", "default")
    if sampling_mode == "balanced_4block":
        traindataset, train_batch_sampler, testdataset, block_info = build_4block_balanced_train_and_test_datasets(
            tokenizer,
            data_cfg,
            max_len=training_parameters["max_len"],
            target_len=data_cfg["block_size"] * data_cfg["block_num"],
            seed=SEED,
            batch_size=training_parameters["bs"],
        )
        print("[Data] using balanced_4block sampling with train block sizes:", block_info)
    else:
        train_batch_sampler = None
        traindataset = build_dataset_rank(
            tokenizer,
            data_cfg["train_dataset"],
            training_parameters["max_len"],
            target_len=data_cfg["block_size"] * data_cfg["block_num"],
            get_test_subset=False,
            **dataset_common_kwargs,
        )
        testdataset = build_dataset_rank(
            tokenizer,
            data_cfg["train_dataset"],
            training_parameters["max_len"],
            target_len=data_cfg["block_size"] * data_cfg["block_num"],
            get_test_subset=True,
            **dataset_common_kwargs,
        )
    print(f"Train data: {len(traindataset)}, Test data: {len(testdataset)}")

    if train_batch_sampler is None:
        train_loader = DataLoader(
            traindataset,
            batch_size=training_parameters["bs"],
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            collate_fn=DataCollatorWithPaddingV2(block_size=train_config["data"]["block_size"], block_num=train_config["data"]["block_num"])
        )
    else:
        train_loader = DataLoader(
            traindataset,
            batch_sampler=train_batch_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=DataCollatorWithPaddingV2(block_size=train_config["data"]["block_size"], block_num=train_config["data"]["block_num"])
        )
    test_loader = DataLoader(
        testdataset,
        batch_size=training_parameters["bs"],
        num_workers=0,
        pin_memory=True,
        collate_fn=DataCollatorWithPaddingV2(block_size=train_config["data"]["block_size"], block_num=train_config["data"]["block_num"])
    )

    # -------------------------
    # Optimizer/Scheduler (PyTorch equivalent of DS WarmupDecayLR)
    # -------------------------
    # IMPORTANT: train_config optimizer lr is 0; in PyTorch set base lr = warmup_max_lr
    warmup_steps = int(train_config["scheduler"]["params"]["warmup_num_steps"])
    total_steps  = int(train_config["scheduler"]["params"]["total_num_steps"])
    lr_dict      = train_config["optimizer"]["params"]["lr"]

    betas = tuple(train_config["optimizer"]["params"]["betas"])
    weight_decay = train_config["optimizer"]["params"]["weight_decay"]

    talk_params, lora_params, lm_head_params, other_params = split_params(model)
    print(f"talk trainable params: {sum(p.numel() for p in talk_params):,}")
    print(f"lora trainable params: {sum(p.numel() for p in lora_params):,}")
    print(f"lm_head trainable params: {sum(p.numel() for p in lm_head_params):,}")
    print(f"other trainable params: {sum(p.numel() for p in other_params):,}")
    if train_config["train_lm_head"]["enabled"] and len(lm_head_params) == 0:
        print("[WARN] train_lm_head is enabled but no talk_lm_head parameters were found.")
    
    param_groups = [{"name": "talk", "params": talk_params + other_params, "lr": lr_dict['talk'], "weight_decay": weight_decay["talk"]}]
    if train_config["lora"]["enabled"]:
        param_groups.append({"name": "lora", "params": lora_params, "lr": lr_dict["lora"], "weight_decay": weight_decay["lora"]})
    if train_config["train_lm_head"]["enabled"]:
        param_groups.append({"name": "lm_head", "params": lm_head_params, "lr": lr_dict["lm_head"], "weight_decay": weight_decay["lm_head"]})
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"[OPT] group={g['name']}, lr={g['lr']}, wd={g['weight_decay']}, params={n_params:,}")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=betas,
    )

    grad_accum = int(train_config.get("gradient_accumulation_steps", 1))
    grad_clip = float(train_config.get("gradient_clipping", 0.0) or 0.0)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Load ckpt AFTER optimizer/scheduler exists, BEFORE accelerator.prepare so the
    # raw module's state_dict matches the saved keys.
    if ckpt_dir is not None:
        start_epoch = load_ckpt(ckpt_dir, model, optimizer, scheduler, map_location="cpu")
    else:
        warm_start_dir = resolve_state_dir(init_from_state_dir_cfg, filename="ckpt.pt")
        if warm_start_dir is not None:
            _ = load_ckpt(
                warm_start_dir,
                model,
                optimizer=None,
                scheduler=None,
                map_location="cpu",
                strict_talk=False,
            )
            start_epoch = 0
            print(f"[CKPT] warm-started talk/LoRA weights from: {warm_start_dir} (strict_talk=False)")
        elif init_from_state_dir_cfg:
            print(f"[CKPT] init_from_state_dir not found/invalid: {init_from_state_dir_cfg}; training from scratch.")
        else:
            print("[CKPT] no previous checkpoint found; training from scratch.")

    # Wrap model / optimizer / dataloaders / scheduler with Accelerate. In single-GPU
    # mode this is mostly a no-op; in 2-GPU DDP mode this wires up gradient sync and
    # shards the dataloader. `model_raw` keeps a handle on the unwrapped module so we
    # can reach T3-specific attributes (length, talk_embed_weight, talk_model, ...)
    # without going through the DDP wrapper.
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    model_raw = accelerator.unwrap_model(model)

    num_epochs = int(training_parameters["num_epochs"])

    # -------------------------
    # WandB (optional)
    # -------------------------
    use_wandb = accelerator.is_main_process
    if use_wandb:
        import wandb
        # Do NOT hardcode keys; set WANDB_API_KEY in environment.
        wandb.login(key="")
        wandb.init(project="TalkingMachine", name=f'{training_parameters["wandb_name"]}-{train_config["data"]["block_size"]}-{train_config["data"]["block_num"]}-{train_config["gradient_accumulation_steps"]}', config=train_config)

    # -------------------------
    # Train/Test loops
    # -------------------------
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    def run_epoch(loader, train: bool, finetune: bool, epoch_num: int):
        nonlocal global_step
        if train:
            model.train()
        else:
            model.eval()

        loss_cfg = train_config.get("loss", {})
        # Per-iter stats inside the denoise loop (argmax / log_softmax over a
        # full vocab) cost ~5-15% of step time but are telemetry-only. Compute
        # them every N opt steps; the live training loss path stays untouched.
        # Default 1 = compute every step (bit-identical to the pre-gate behaviour).
        step_metrics_every_n = max(1, int(loss_cfg.get("step_metrics_every_n_steps", 1)))

        epoch_acces = [[] for _ in range(model_raw.length)]
        epoch_plosses = [[] for _ in range(model_raw.length)]
        epoch_certain_losses = [[] for _ in range(model_raw.length)]
        epoch_certain_accs = [[] for _ in range(model_raw.length)]
        epoch_decode_losses = [[] for _ in range(model_raw.length)]
        epoch_decode_accs = [[] for _ in range(model_raw.length)]
        epoch_clip_flags = []
        epoch_grad_norm_pre = []
        epoch_grad_norm_post = []

        batch_acces = [[] for _ in range(model_raw.length)]
        batch_plosses = [[] for _ in range(model_raw.length)]
        batch_certain_losses = [[] for _ in range(model_raw.length)]
        batch_certain_accs = [[] for _ in range(model_raw.length)]
        batch_decode_losses = [[] for _ in range(model_raw.length)]
        batch_decode_accs = [[] for _ in range(model_raw.length)]
        batch_clip_flags = []
        batch_grad_norm_pre = []
        batch_grad_norm_post = []

        pbar = tqdm(loader, desc=("train" if train else "test"), disable=not accelerator.is_local_main_process)
        for batch_idx, data in enumerate(pbar):
            do_vis = (train and (global_step % VIS_EVERY_OPT_STEP == 0) and ((batch_idx) % grad_accum == 0))
            # Whether to compute per-iter telemetry (argmax/log_softmax) this batch.
            # Constant across the grad_accum window because global_step only ticks
            # on the optimizer-step boundary.
            compute_step_metrics = (global_step % step_metrics_every_n == 0)

            # accelerator.prepare(loader) puts batch tensors on `device` already.
            input_ids_think = data["input_ids"]
            pos_ids_think   = data["position_ids"]
            attn_think      = data["attention_mask"]
            block_mask_think = get_block_mask_for_batch(data, device)
            # -----------------
            # Think forward
            # -----------------
            ctx = torch.enable_grad() if (train and finetune) else torch.no_grad()
            with ctx:
                think_outputs = model(
                    input_ids=input_ids_think,
                    position_ids=pos_ids_think,
                    attention_mask=attn_think,
                    block_mask=block_mask_think,
                    use_cache=False,
                    output_hidden_states=True
                )
            think_rps = think_outputs.hidden_states  # [B, S+C, H]
            B = think_rps.size(0)
            H = think_rps.size(-1)

            # plosses = []
            loss_sums = []
            counts = []
            acces = []
            certain_losses = []
            certain_accs = []
            decode_losses = []
            decode_accs = []

            # -----------------
            # Talking iterations
            # -----------------
            target_talk = data["target"].view(-1, model_raw.length)  # [B*G, L]
            mask_bool_talk = data["loss_mask"].bool()                # [B, S+C]
            input_ids = input_ids_think[mask_bool_talk].view(data["input_ids"].size(0), -1)
            input_ids = input_ids.view(-1, model_raw.length)         # [B*G, L]
            rps = think_rps[mask_bool_talk].view(B, -1, H)           # [B, L, H]
            rps = rps.view(-1, model_raw.length, rps.shape[-1])      # [B*G, L, H]
            talk_attention_mask = torch.ones_like(input_ids, dtype=torch.long).view(-1, model_raw.length)
            # Talk model is unmasked (talk_attention_mask is all-ones; talk-side
            # LLaDABlock.attention discards attention_bias). Pass nothing so the
            # FlashAttention no-mask fast path is taken.
            loss_mask = torch.ones_like(target_talk, dtype=torch.float32).view(-1, model_raw.length)

            if do_vis:
                input_ids_init_BG_L = input_ids.detach().clone()
                steps_pred_BG_L = []
                steps_mask_BG_L = []

            # If eval: keep everything in no_grad()
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                input_embeds = F.embedding(input_ids, model_raw.talk_embed_weight)  # initial emb (step 0)
                for idx in range(model_raw.length):
                    if not loss_mask.bool().any():
                        break
                    talk_outputs = model(
                        input_ids=None,
                        inputs_embeds=input_embeds,
                        inputs_repres=rps,
                        attention_mask=talk_attention_mask,
                        use_cache=False,
                        output_hidden_states=True
                    )
                    logits = talk_outputs.logits.float()
                    rps = talk_outputs.hidden_states

                    loss_sum_i, count_i = per_step_loss_sum_and_count(
                        loss_cfg=loss_cfg,
                        logits=logits,
                        target=target_talk,
                        loss_mask=loss_mask,
                    )
                    loss_sums.append(loss_sum_i)
                    counts.append(count_i)

                    # Per-iter telemetry. Each block here runs a full-vocab
                    # argmax / log_softmax over [BG, L, V] and these aren't on
                    # the training path — gated behind step_metrics_every_n_steps.
                    if compute_step_metrics:
                        correct, total = calculate_correct_counts(logits, target_talk, loss_mask)
                        certain_cols, certain_valid = pick_single_position(
                            loss_mask=loss_mask,
                            logits=logits,
                            mode="greedy",
                        )
                        certain_loss_i, certain_acc_i = selected_position_loss_acc(
                            logits=logits,
                            target=target_talk,
                            cols=certain_cols,
                            valid=certain_valid,
                        )
                        acces.append(correct / max(1, total))
                        certain_losses.append(certain_loss_i)
                        certain_accs.append(certain_acc_i)

                    denoise_cfg = train_config["denoise"]
                    reveal_cfg = get_denoise_reveal_config(denoise_cfg)
                    decode_cfg = get_denoise_decode_config(denoise_cfg, default_k=reveal_cfg["k"])

                    if do_vis and idx < VIS_MAX_STEPS:
                        steps_pred_BG_L.append(logits.argmax(dim=-1).detach())  # [B*G, L]
                        steps_mask_BG_L.append(loss_mask.detach())             # [B*G, L]

                    # denoise step updates input_ids + loss_mask
                    loss_mask_prev = loss_mask
                    if train_config["soft_inputs"]["enabled"]:
                        soft_cfg = train_config["soft_inputs"]
                        kwargs = dict(
                            input_ids=input_ids,
                            target=target_talk,
                            loss_mask=loss_mask,
                            logits=logits,
                            emb_weight=model_raw.talk_embed_weight,
                            k_reveal=reveal_cfg["k"],
                            soft_topk=soft_cfg["top_k"],
                            soft_temp=soft_cfg["temperature"],
                            mode=reveal_cfg.get("policy", "random"),
                            decode_cfg=decode_cfg,
                        )
                        # Only enable mask-mix when user explicitly sets lam_max/lam_min in config
                        if ("lam_max" in soft_cfg) or ("lam_min" in soft_cfg):
                            kwargs["lam_max"] = float(soft_cfg.get("lam_max", 0.7))  # sensible default once enabled
                            kwargs["lam_min"] = float(soft_cfg.get("lam_min", 0.0))
                            mid = getattr(tokenizer, "mask_token_id", None)
                            if mid is not None:
                                kwargs["mask_token_id"] = mid

                        input_ids, input_embeds, loss_mask = denoise_k_step_soft_embed_v2(**kwargs)
                    else:
                        input_ids, loss_mask = denoise_k_step(
                            input_ids=input_ids,
                            target=target_talk,
                            loss_mask=loss_mask,
                            logits=logits,
                            k=reveal_cfg["k"],
                            mode=reveal_cfg.get("policy", "random"),
                            decode_cfg=decode_cfg,
                        )
                        input_embeds = F.embedding(input_ids, model_raw.talk_embed_weight)

                    if compute_step_metrics:
                        revealed_mask = loss_mask_prev.bool() & (~loss_mask.bool())
                        decode_loss_i, decode_acc_i = masked_position_loss_acc(
                            logits=logits,
                            target=target_talk,
                            position_mask=revealed_mask,
                        )
                        decode_losses.append(decode_loss_i)
                        decode_accs.append(decode_acc_i)

                    if train_config["detach_recurrence"]["enabled"] and ((idx + 1) % train_config["detach_recurrence"]["every_r_steps"] == 0):
                        rps, input_embeds = detach_state(rps, input_embeds)

                w = build_step_weights(loss_cfg, num_steps=len(loss_sums), epoch=epoch_num).to(device)

                weighted_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                weighted_count    = torch.zeros((), device=device, dtype=torch.float32)

                for i in range(len(loss_sums)):
                    weighted_loss_sum = weighted_loss_sum + w[i] * loss_sums[i]
                    weighted_count    = weighted_count    + w[i] * counts[i]

                clamp_min = float(loss_cfg.get("normalize", {}).get("count_clamp_min", 1e-6))
                loss = weighted_loss_sum / weighted_count.clamp_min(clamp_min)

                if train:
                    # `accelerator.accumulate` matches the manual `loss / grad_accum`
                    # pattern: it scales the loss internally and only triggers DDP
                    # gradient sync on the boundary step.
                    accelerator.backward(loss / grad_accum)

                    if (batch_idx + 1) % grad_accum == 0:
                        clip_flag = 0.0
                        pre_norm = 0.0
                        post_norm = 0.0
                        if grad_clip > 0:
                            total_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                            pre_norm = float(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
                            post_norm = float(min(pre_norm, grad_clip))
                            clip_flag = 1.0 if pre_norm > grad_clip else 0.0
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        batch_clip_flags.append(clip_flag)
                        batch_grad_norm_pre.append(pre_norm)
                        batch_grad_norm_post.append(post_norm)

            # record batch stats (move scalar to cpu)
            for i in range(len(acces)):
                batch_acces[i].append(acces[i])
            # for i in range(len(plosses)):
            #     batch_plosses[i].append(plosses[i].detach().item())
            for i in range(len(loss_sums)):
                batch_plosses[i].append(loss_sums[i].detach().item() / counts[i].detach().item())
            for i in range(len(certain_losses)):
                batch_certain_losses[i].append(certain_losses[i])
                batch_certain_accs[i].append(certain_accs[i])
                batch_decode_losses[i].append(decode_losses[i])
                batch_decode_accs[i].append(decode_accs[i])

            if (batch_idx + 1) % grad_accum == 0:
                # Always-on stats (cheap; based on data we collect every step).
                batch_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_plosses]
                batch_clip_rate = float(np.mean(batch_clip_flags)) if len(batch_clip_flags) else 0.0
                batch_grad_pre_mean = float(np.mean(batch_grad_norm_pre)) if len(batch_grad_norm_pre) else 0.0
                batch_grad_post_mean = float(np.mean(batch_grad_norm_post)) if len(batch_grad_norm_post) else 0.0
                for i in range(len(batch_loss_mean)):
                    epoch_plosses[i].append(batch_loss_mean[i])
                if len(batch_clip_flags):
                    epoch_clip_flags.extend(batch_clip_flags)
                    epoch_grad_norm_pre.extend(batch_grad_norm_pre)
                    epoch_grad_norm_post.extend(batch_grad_norm_post)

                # Gated stats: only computed when compute_step_metrics; the
                # underlying batch_* lists are empty otherwise.
                if compute_step_metrics:
                    batch_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_acces]
                    batch_certain_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_certain_losses]
                    batch_certain_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_certain_accs]
                    batch_decode_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_decode_losses]
                    batch_decode_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_decode_accs]
                    for i in range(len(batch_acc_mean)):
                        epoch_acces[i].append(batch_acc_mean[i])
                    for i in range(len(batch_certain_loss_mean)):
                        epoch_certain_losses[i].append(batch_certain_loss_mean[i])
                        epoch_certain_accs[i].append(batch_certain_acc_mean[i])
                        epoch_decode_losses[i].append(batch_decode_loss_mean[i])
                        epoch_decode_accs[i].append(batch_decode_acc_mean[i])

                # Reset all per-batch accumulators (the gated-off ones are
                # already empty, so resetting is a no-op for them).
                batch_acces = [[] for _ in range(model_raw.length)]
                batch_plosses = [[] for _ in range(model_raw.length)]
                batch_certain_losses = [[] for _ in range(model_raw.length)]
                batch_certain_accs = [[] for _ in range(model_raw.length)]
                batch_decode_losses = [[] for _ in range(model_raw.length)]
                batch_decode_accs = [[] for _ in range(model_raw.length)]
                batch_clip_flags = []
                batch_grad_norm_pre = []
                batch_grad_norm_post = []

                # wandb step logs.
                if use_wandb and train:
                    logdict = {
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/clip_rate": batch_clip_rate,
                        "train/grad_norm_preclip": batch_grad_pre_mean,
                        "train/grad_norm_postclip": batch_grad_post_mean,
                    }
                    for i, v in enumerate(batch_loss_mean):
                        logdict[f"train/ploss_{i:02d}"] = float(v)
                    if compute_step_metrics:
                        for i, a in enumerate(batch_acc_mean):
                            logdict[f"train/acc_{i:02d}"] = float(a)
                        for i, v in enumerate(batch_certain_loss_mean):
                            logdict[f"train/certain_loss_{i:02d}"] = float(v)
                        for i, a in enumerate(batch_certain_acc_mean):
                            logdict[f"train/certain_acc_{i:02d}"] = float(a)
                        for i, v in enumerate(batch_decode_loss_mean):
                            logdict[f"train/decode_loss_{i:02d}"] = float(v)
                        for i, a in enumerate(batch_decode_acc_mean):
                            logdict[f"train/decode_acc_{i:02d}"] = float(a)
                    wandb.log(logdict)
                if train:
                    record = {
                        "epoch": int(epoch_num),
                        "global_step": int(global_step),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "loss_by_iter": batch_loss_mean,
                        "loss_mean": float(np.mean(batch_loss_mean)) if batch_loss_mean else 0.0,
                        "clip_rate": batch_clip_rate,
                        "grad_norm_preclip": batch_grad_pre_mean,
                        "grad_norm_postclip": batch_grad_post_mean,
                        "decode_mode": get_policy_label(
                            get_denoise_reveal_config(train_config["denoise"]).get("policy"),
                            "greedy",
                        ),
                    }
                    if compute_step_metrics:
                        record.update({
                            "acc_by_iter": batch_acc_mean,
                            "certain_loss_by_iter": batch_certain_loss_mean,
                            "certain_acc_by_iter": batch_certain_acc_mean,
                            "decode_loss_by_iter": batch_decode_loss_mean,
                            "decode_acc_by_iter": batch_decode_acc_mean,
                            "acc_mean": float(np.mean(batch_acc_mean)),
                            "certain_loss_mean": float(np.mean(batch_certain_loss_mean)),
                            "certain_acc_mean": float(np.mean(batch_certain_acc_mean)),
                            "decode_loss_mean": float(np.mean(batch_decode_loss_mean)),
                            "decode_acc_mean": float(np.mean(batch_decode_acc_mean)),
                        })
                    append_jsonl(STEP_METRIC_LOG, record)



            if do_vis:
                visualize_t3_batch_trace(
                    tokenizer=tokenizer,
                    data=data,
                    block_size=train_config["data"]["block_size"],
                    block_num=train_config["data"]["block_num"],
                    bidx=VIS_BIDX,
                    target_BG_L=target_talk,                 # [B*G, L]
                    input_ids_init_BG_L=input_ids_init_BG_L, # [B*G, L]
                    steps_pred_BG_L=steps_pred_BG_L,
                    steps_mask_BG_L=steps_mask_BG_L,
                    max_ctx_tokens=120,
                    max_pairs_per_block=64,
                    max_steps=VIS_MAX_STEPS,
                    tag=f"epoch_batch={batch_idx} opt_step={global_step}",
                    log_path="logs/t3_trace.txt",
                )

        # epoch summary
        epoch_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in epoch_acces]
        epoch_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in epoch_plosses]
        epoch_certain_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in epoch_certain_losses]
        epoch_certain_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in epoch_certain_accs]
        epoch_decode_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in epoch_decode_losses]
        epoch_decode_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in epoch_decode_accs]
        epoch_clip_rate = float(np.mean(epoch_clip_flags)) if len(epoch_clip_flags) else 0.0
        epoch_grad_pre_mean = float(np.mean(epoch_grad_norm_pre)) if len(epoch_grad_norm_pre) else 0.0
        epoch_grad_post_mean = float(np.mean(epoch_grad_norm_post)) if len(epoch_grad_norm_post) else 0.0
        return {
            "acc_by_iter": epoch_acc_mean,
            "loss_by_iter": epoch_loss_mean,
            "certain_loss_by_iter": epoch_certain_loss_mean,
            "certain_acc_by_iter": epoch_certain_acc_mean,
            "decode_loss_by_iter": epoch_decode_loss_mean,
            "decode_acc_by_iter": epoch_decode_acc_mean,
            "clip_rate": epoch_clip_rate,
            "grad_norm_preclip": epoch_grad_pre_mean,
            "grad_norm_postclip": epoch_grad_post_mean,
        }

    # -------------------------
    # Main epoch loop
    # -------------------------
    for epoch in range(start_epoch, num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        # Train
        train_stats = run_epoch(train_loader, train=True, finetune=True, epoch_num=epoch)

        print("Train:")
        if use_wandb:
            import wandb
            wandb.log(
                {
                    "train/epoch_clip_rate": train_stats["clip_rate"],
                    "train/epoch_grad_norm_preclip": train_stats["grad_norm_preclip"],
                    "train/epoch_grad_norm_postclip": train_stats["grad_norm_postclip"],
                }
            )
        print(
            f"  clip: rate {train_stats['clip_rate']*100:5.2f}% | "
            f"grad pre {train_stats['grad_norm_preclip']:.4f} | "
            f"grad post {train_stats['grad_norm_postclip']:.4f}"
        )
        for i in range(len(train_stats["acc_by_iter"])):
            if use_wandb:
                import wandb
                wandb.log(
                    {
                        f"train/epochacc_{i:02d}": train_stats["acc_by_iter"][i],
                        f"train/epochploss_{i:02d}": train_stats["loss_by_iter"][i],
                        f"train/epoch_certain_loss_{i:02d}": train_stats["certain_loss_by_iter"][i],
                        f"train/epoch_certain_acc_{i:02d}": train_stats["certain_acc_by_iter"][i],
                        f"train/epoch_decode_loss_{i:02d}": train_stats["decode_loss_by_iter"][i],
                        f"train/epoch_decode_acc_{i:02d}": train_stats["decode_acc_by_iter"][i],
                    }
                )
            print(
                f"  iter {i}: Acc {train_stats['acc_by_iter'][i]*100:5.2f}% | "
                f"pLoss {train_stats['loss_by_iter'][i]:.6f} | "
                f"certain Acc {train_stats['certain_acc_by_iter'][i]*100:5.2f}% | "
                f"certain Loss {train_stats['certain_loss_by_iter'][i]:.6f} | "
                f"decode Acc {train_stats['decode_acc_by_iter'][i]*100:5.2f}% | "
                f"decode Loss {train_stats['decode_loss_by_iter'][i]:.6f}"
            )

        # Test
        test_stats = run_epoch(test_loader, train=False, finetune=False, epoch_num=epoch)

        print("Test:")
        if use_wandb:
            import wandb
            wandb.log(
                {
                    "test/epoch_clip_rate": test_stats["clip_rate"],
                    "test/epoch_grad_norm_preclip": test_stats["grad_norm_preclip"],
                    "test/epoch_grad_norm_postclip": test_stats["grad_norm_postclip"],
                }
            )
        print(
            f"  clip: rate {test_stats['clip_rate']*100:5.2f}% | "
            f"grad pre {test_stats['grad_norm_preclip']:.4f} | "
            f"grad post {test_stats['grad_norm_postclip']:.4f}"
        )
        for i in range(len(test_stats["acc_by_iter"])):
            if use_wandb:
                import wandb
                wandb.log(
                    {
                        f"test/epochacc_{i:02d}": test_stats["acc_by_iter"][i],
                        f"test/epochploss_{i:02d}": test_stats["loss_by_iter"][i],
                        f"test/epoch_certain_loss_{i:02d}": test_stats["certain_loss_by_iter"][i],
                        f"test/epoch_certain_acc_{i:02d}": test_stats["certain_acc_by_iter"][i],
                        f"test/epoch_decode_loss_{i:02d}": test_stats["decode_loss_by_iter"][i],
                        f"test/epoch_decode_acc_{i:02d}": test_stats["decode_acc_by_iter"][i],
                    }
                )
            print(
                f"  iter {i}: Acc {test_stats['acc_by_iter'][i]*100:5.2f}% | "
                f"pLoss {test_stats['loss_by_iter'][i]:.6f} | "
                f"certain Acc {test_stats['certain_acc_by_iter'][i]*100:5.2f}% | "
                f"certain Loss {test_stats['certain_loss_by_iter'][i]:.6f} | "
                f"decode Acc {test_stats['decode_acc_by_iter'][i]*100:5.2f}% | "
                f"decode Loss {test_stats['decode_loss_by_iter'][i]:.6f}"
            )
        append_jsonl(
            EPOCH_METRIC_LOG,
            {
                "epoch": int(epoch),
                "train": train_stats,
                "test": test_stats,
                "decode_mode": get_policy_label(
                    get_denoise_reveal_config(train_config["denoise"]).get("policy"),
                    "greedy",
                ),
            },
        )

        # clear cache
        torch.cuda.empty_cache()

        # Save PyTorch checkpoint (talking_ml only + optim/sched).
        # `accelerator.wait_for_everyone()` syncs ranks before save; only the main
        # process writes. The unwrapped state_dict matches the pre-DDP key layout,
        # which keeps existing `load_ckpt` callers working.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_ckpt(
                args.savedir,
                epoch,
                model_raw,
                optimizer,
                scheduler,
                extra={
                    "device": str(device),
                    "global_step": global_step,
                },
                model_config=model_config
            )
            state_dir = os.path.join(args.savedir, f"state_{epoch}")
            run_inference_suite(
                train_config=train_config,
                savedir=args.savedir,
                state_dir=state_dir,
                epoch=epoch,
                device=str(device),
            )
        accelerator.wait_for_everyone()

    print("\nDone.")


if __name__ == "__main__":
    main()

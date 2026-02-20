# -*- coding: utf-8 -*-
"""
Single-process, pure PyTorch training:
- thought model (frozen, no_grad) on THINK_DEVICE
- talking_ml (trainable) on TALK_DEVICE
- manual warmup+linear decay LR scheduler (DeepSpeed WarmupDecayLR equivalent)
- safe device handling for masks / indexing
- PyTorch-style checkpoint saving (talk_model + optimizer + scheduler + meta)
"""

import os, re, json
import argparse
import math
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate.utils import set_seed
from tqdm import tqdm

from model.modeling_t3 import T3Model
from train.data_process import build_dataset_rank, DataCollatorWithPadding, DataCollatorWithPaddingV2
from train.visualize import visualize_t3_batch_trace
from utils import AttrDict, load_ckpt, save_ckpt, topk_soft_embedding_from_logits, denoise_k_step_soft_embed_v2


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

def append_jsonl(path: str, payload: dict):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def denoise_k_step(input_ids, target, loss_mask, k=1, generator=None):
    """
    input_ids: [BG, L] on TALK_DEVICE
    target:    [BG, L] on TALK_DEVICE
    loss_mask: [BG, L] on TALK_DEVICE (0/1 float or bool)
    """
    device = input_ids.device
    B, C = input_ids.shape

    active = loss_mask.bool()
    if generator is not None:
        assert generator.device == device
        scores = torch.rand((B, C), device=device, generator=generator)
    else:
        scores = torch.rand((B, C), device=device)
    scores = scores.masked_fill(~active, float("-inf"))

    idx = scores.topk(k=min(k, C), dim=1).indices  # [B, k]
    chosen_active = active.gather(1, idx)          # [B, k]

    rows = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)
    input_ids = input_ids.clone()
    loss_mask = loss_mask.clone()

    rows = rows[chosen_active]
    cols = idx[chosen_active]

    input_ids[rows, cols] = target[rows, cols]
    loss_mask[rows, cols] = 0

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

def split_params(model):
    talk_params = []
    lora_params = []
    lm_head_params = []
    other_trainable = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        if ("lm_head" in nl) or ("ff_out" in nl):
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
    parser.add_argument("--think_device1", type=str, default="cuda:2")
    parser.add_argument("--think_device2", type=str, default="cuda:3")
    parser.add_argument("--talk_device", type=str, default="cuda:1")
    args = parser.parse_args()
    
    THINK_DEVICE1 = args.think_device1
    THINK_DEVICE2 = args.think_device2
    TALK_DEVICE = args.talk_device
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
    if train_config["lora"]["enabled"]:
        model_config['lora'] = train_config["lora"]
    if train_config['rps_residual']['enabled']:
        model_config['rps_residual'] = train_config['rps_residual']
    if train_config['soft_inputs']['enabled']:
        model_config['soft_inputs'] = train_config['soft_inputs']
    # -------------------------
    # Model
    # -------------------------
    if args.savedir == "0":
        args.savedir = f'{training_parameters["wandb_name"]}-{train_config["data"]["block_size"]}-{train_config["data"]["block_num"]}-{train_config["gradient_accumulation_steps"]}'
    os.makedirs(args.savedir, exist_ok=True)

    model = T3Model(model_config, think_dev1=THINK_DEVICE1, think_dev2=THINK_DEVICE2, talk_dev=TALK_DEVICE)

    freeze_parameters(model)

    # Eval/train modes
    model.train()
    model.think_model.train()
    model.talk_model.train()

    # If no ckpt, init talking
    ckpt_dir, start_epoch = find_max_state_with_file(args.savedir, filename="ckpt.pt")

    # -------------------------
    # Tokenizer / Data
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_model_name_or_path"], trust_remote_code=True)

    traindataset = build_dataset_rank(
        tokenizer, train_config["data"]["train_dataset"], training_parameters["max_len"], target_len=train_config["data"]["block_size"]*train_config["data"]["block_num"], splits=args.split,
        get_test_subset=False, seed=SEED
    )
    testdataset = build_dataset_rank(
        tokenizer, train_config["data"]["train_dataset"], training_parameters["max_len"], target_len=train_config["data"]["block_size"]*train_config["data"]["block_num"], splits=args.split,
        get_test_subset=True, seed=SEED
    )
    print(f"Train data: {len(traindataset)}, Test data: {len(testdataset)}")

    train_loader = DataLoader(
        traindataset,
        batch_size=training_parameters["bs"],
        num_workers=1,
        pin_memory=True,
        collate_fn=DataCollatorWithPaddingV2(block_size=train_config["data"]["block_size"], block_num=train_config["data"]["block_num"])
    )
    test_loader = DataLoader(
        testdataset,
        batch_size=training_parameters["bs"],
        num_workers=1,
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
    
    param_groups = [{"name": "talk", "params": talk_params + other_params, "lr": lr_dict['talk'], "weight_decay": weight_decay["talk"]}]
    if train_config["lora"]["enabled"]:
        param_groups.append({"name": "lora", "params": lora_params, "lr": lr_dict["lora"], "weight_decay": weight_decay["lora"]})
    if train_config["train_lm_head"]["enabled"]:
        param_groups.append({"name": "lm_head", "params": lm_head_params, "lr": lr_dict["lm_head"], "weight_decay": weight_decay["lm_head"]})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=betas,
    )
    
    grad_accum = int(train_config.get("gradient_accumulation_steps", 1))
    grad_clip = float(train_config.get("gradient_clipping", 0.0) or 0.0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Load ckpt AFTER optimizer/scheduler exists
    if ckpt_dir is not None:
        # load ckpt (map to CPU first; state_dict copy to correct device tensors is handled by optimizer load)
        start_epoch = load_ckpt(ckpt_dir, model, optimizer, scheduler, map_location="cpu")

    num_epochs = int(training_parameters["num_epochs"])

    # -------------------------
    # WandB (optional)
    # -------------------------
    use_wandb = True
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

        epoch_acces = [[] for _ in range(model.length)]
        epoch_plosses = [[] for _ in range(model.length)]
        epoch_certain_losses = [[] for _ in range(model.length)]
        epoch_certain_accs = [[] for _ in range(model.length)]
        epoch_decode_losses = [[] for _ in range(model.length)]
        epoch_decode_accs = [[] for _ in range(model.length)]

        batch_acces = [[] for _ in range(model.length)]
        batch_plosses = [[] for _ in range(model.length)]
        batch_certain_losses = [[] for _ in range(model.length)]
        batch_certain_accs = [[] for _ in range(model.length)]
        batch_decode_losses = [[] for _ in range(model.length)]
        batch_decode_accs = [[] for _ in range(model.length)]
        
        pbar = tqdm(loader, desc=("train" if train else "test"))
        for batch_idx, data in enumerate(pbar):
            do_vis = (train and (global_step % VIS_EVERY_OPT_STEP == 0) and ((batch_idx) % grad_accum == 0))

            # Thought inputs on THINK_DEVICE
            input_ids_think = data["input_ids"].to(THINK_DEVICE1, non_blocking=True)
            pos_ids_think = data["position_ids"].to(THINK_DEVICE1, non_blocking=True)
            attn_think      = data["attention_mask"].to(THINK_DEVICE1, non_blocking=True)
            bias_think      = data["attention_bias"].to(THINK_DEVICE1, non_blocking=True)
            # for it in input_ids_think:
            #     print(tokenizer.decode(it.detach().view(-1).tolist()))
            # -----------------
            # Think forward
            # -----------------
            ctx = torch.enable_grad() if (train and finetune) else torch.no_grad()
            with ctx:
                think_outputs = model(
                    input_ids=input_ids_think,
                    position_ids=pos_ids_think,
                    attention_mask=attn_think,
                    attention_bias=bias_think,
                    use_cache=False,
                    output_hidden_states=True
                )
            think_rps = think_outputs.hidden_states  # [B, S+C, H] on THINK_DEVICE
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
            # Target on TALK_DEVICE (assumed [B, L] == [B, C])
            target_talk = data["target"].to(TALK_DEVICE, non_blocking=True)
            target_talk = target_talk.view(-1, model.length)  #[B * G, L]
            # Build device-specific masks
            mask_bool_talk = data["loss_mask"].to(TALK_DEVICE, non_blocking=True).bool()  # [B, S+C]
            # Build talk input_ids on TALK_DEVICE using talk mask
            input_ids = input_ids_think.to(TALK_DEVICE)[mask_bool_talk].view(data["input_ids"].size(0), -1)  # [B, L]
            input_ids = input_ids.view(-1, model.length) #[B * G, L]
            # Move rps once to TALK_DEVICE
            rps = think_rps.to(TALK_DEVICE, non_blocking=True)
            rps = rps[mask_bool_talk].view(B, -1, H)  # [B, L, H]
            rps = rps.view(-1, model.length, rps.shape[-1])  #[B * G, L]
            # Build attention mask for talk (all ones) on TALK_DEVICE
            talk_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=TALK_DEVICE)
            talk_attention_mask = talk_attention_mask.view(-1, model.length)  #[B * G, L]
            talk_bias = torch.zeros((1, 1, model.length, model.length), device=TALK_DEVICE, dtype=torch.float32)
            # loss mask on TALK_DEVICE, same shape as target_talk ([B, L])
            loss_mask = torch.ones_like(target_talk, dtype=torch.float32, device=TALK_DEVICE)
            loss_mask = loss_mask.view(-1, model.length)
            
            if do_vis:
                input_ids_init_BG_L = input_ids.detach().clone()
                steps_pred_BG_L = []
                steps_mask_BG_L = []

            # If eval: keep everything in no_grad()
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                input_embeds = F.embedding(input_ids, model.talk_embed_weight)  # initial emb (step 0)
                for idx in range(model.length):
                    talk_outputs = model(
                        input_ids=None,
                        inputs_embeds=input_embeds,
                        inputs_repres=rps,
                        attention_mask=talk_attention_mask,
                        attention_bias=talk_bias,
                        use_cache=False,
                        output_hidden_states=True
                    )
                    logits = talk_outputs.logits.float()
                    rps = talk_outputs.hidden_states

                    # out_logp = F.log_softmax(logits, dim=-1)

                    # loss_i = calculate_ploss(out_logp, target_talk, loss_mask)
                    # loss_sum_i, count_i = ploss_sum_and_count(out_logp, target_talk, loss_mask)
                    loss_sum_i, count_i = per_step_loss_sum_and_count(
                        loss_cfg=loss_cfg,
                        logits=logits,
                        target=target_talk,
                        loss_mask=loss_mask,
                    )

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
                    decode_cols, decode_valid = pick_single_position(
                        loss_mask=loss_mask,
                        logits=logits,
                        mode=train_config["denoise"]["reveal_strategy"],
                    )
                    decode_loss_i, decode_acc_i = selected_position_loss_acc(
                        logits=logits,
                        target=target_talk,
                        cols=decode_cols,
                        valid=decode_valid,
                    )

                    # plosses.append(loss_i)
                    loss_sums.append(loss_sum_i)
                    counts.append(count_i)
                    acces.append(correct / max(1, total))
                    certain_losses.append(certain_loss_i)
                    certain_accs.append(certain_acc_i)
                    decode_losses.append(decode_loss_i)
                    decode_accs.append(decode_acc_i)
                    
                    if do_vis and idx < VIS_MAX_STEPS:
                        steps_pred_BG_L.append(logits.argmax(dim=-1).detach())  # [B*G, L]
                        steps_mask_BG_L.append(loss_mask.detach())             # [B*G, L]
                    
                    # denoise step updates input_ids + loss_mask (both on TALK_DEVICE)
                    if train_config["soft_inputs"]["enabled"]:
                        soft_cfg = train_config["soft_inputs"]
                        kwargs = dict(
                            input_ids=input_ids,
                            target=target_talk,
                            loss_mask=loss_mask,
                            logits=logits,
                            emb_weight=model.talk_embed_weight,
                            soft_topk=soft_cfg["top_k"],
                            soft_temp=soft_cfg["temperature"],
                            mode=train_config["denoise"]["reveal_strategy"],
                        )
                        # Only enable mask-mix when user explicitly sets lam_max/lam_min in config
                        if ("lam_max" in soft_cfg) or ("lam_min" in soft_cfg):
                            kwargs["lam_max"] = float(soft_cfg.get("lam_max", 0.7))  # sensible default once enabled
                            kwargs["lam_min"] = float(soft_cfg.get("lam_min", 0.0))
                            # also recommend passing mask_token_id explicitly when enabled
                            mid = getattr(tokenizer, "mask_token_id", None)
                            if mid is not None:
                                kwargs["mask_token_id"] = mid
                                
                        input_ids, input_embeds, loss_mask = denoise_k_step_soft_embed_v2(**kwargs)
                    else:
                        input_ids, loss_mask = denoise_k_step(input_ids, target_talk, loss_mask)
                        input_embeds = F.embedding(input_ids, model.talk_embed_weight)  # initial emb (step 0)
                    
                    if train_config["detach_recurrence"]["enabled"] and ((idx + 1) % train_config["detach_recurrence"]["every_r_steps"] == 0):
                        rps, input_embeds = detach_state(rps, input_embeds)
                
                # Version 1
                # ploss_weight = [0.8 ** i if i < 10 else 0.8 ** 10 for i in range(len(plosses))]
                # loss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))
                
                # Version 2
                # w = [0.8 ** i if i < 10 else 0.8 ** 10 for i in range(len(loss_sums))]
                # w[0] *= max(1, 5 - epoch_num)
                # weighted_loss_sum = sum(w[i] * loss_sums[i] for i in range(len(loss_sums)))
                # weighted_count    = sum(w[i] * counts[i]     for i in range(len(counts)))
                # loss = weighted_loss_sum / weighted_count.clamp_min(1e-6)

                # Version 3
                w = build_step_weights(loss_cfg, num_steps=len(loss_sums), epoch=epoch_num).to(TALK_DEVICE)

                weighted_loss_sum = torch.zeros((), device=TALK_DEVICE, dtype=torch.float32)
                weighted_count    = torch.zeros((), device=TALK_DEVICE, dtype=torch.float32)

                for i in range(len(loss_sums)):
                    weighted_loss_sum = weighted_loss_sum + w[i] * loss_sums[i]
                    weighted_count    = weighted_count    + w[i] * counts[i]

                clamp_min = float(loss_cfg.get("normalize", {}).get("count_clamp_min", 1e-6))
                loss = weighted_loss_sum / weighted_count.clamp_min(clamp_min)

                if train:
                    loss = loss / grad_accum
                    loss.backward()
                    
                    if (batch_idx + 1) % grad_accum == 0:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

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
                batch_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_acces]
                batch_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_plosses]
                batch_certain_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_certain_losses]
                batch_certain_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_certain_accs]
                batch_decode_loss_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_decode_losses]
                batch_decode_acc_mean = [float(np.mean(v)) if len(v) else 0.0 for v in batch_decode_accs]
                for i in range(len(batch_acc_mean)):
                    epoch_acces[i].append(batch_acc_mean[i])
                for i in range(len(batch_loss_mean)):
                    epoch_plosses[i].append(batch_loss_mean[i])
                for i in range(len(batch_certain_loss_mean)):
                    epoch_certain_losses[i].append(batch_certain_loss_mean[i])
                    epoch_certain_accs[i].append(batch_certain_acc_mean[i])
                    epoch_decode_losses[i].append(batch_decode_loss_mean[i])
                    epoch_decode_accs[i].append(batch_decode_acc_mean[i])
                batch_acces = [[] for _ in range(model.length)]
                batch_plosses = [[] for _ in range(model.length)]
                batch_certain_losses = [[] for _ in range(model.length)]
                batch_certain_accs = [[] for _ in range(model.length)]
                batch_decode_losses = [[] for _ in range(model.length)]
                batch_decode_accs = [[] for _ in range(model.length)]
                
                # wandb step logs
                if use_wandb and train:
                    logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
                    for i, v in enumerate(batch_loss_mean):
                        logdict[f"train/ploss_{i:02d}"] = float(v)
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
                    append_jsonl(
                        STEP_METRIC_LOG,
                        {
                            "epoch": int(epoch_num),
                            "global_step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_by_iter": batch_loss_mean,
                            "acc_by_iter": batch_acc_mean,
                            "certain_loss_by_iter": batch_certain_loss_mean,
                            "certain_acc_by_iter": batch_certain_acc_mean,
                            "decode_loss_by_iter": batch_decode_loss_mean,
                            "decode_acc_by_iter": batch_decode_acc_mean,
                            "loss_mean": float(np.mean(batch_loss_mean)),
                            "acc_mean": float(np.mean(batch_acc_mean)),
                            "certain_loss_mean": float(np.mean(batch_certain_loss_mean)),
                            "certain_acc_mean": float(np.mean(batch_certain_acc_mean)),
                            "decode_loss_mean": float(np.mean(batch_decode_loss_mean)),
                            "decode_acc_mean": float(np.mean(batch_decode_acc_mean)),
                            "decode_mode": str(train_config["denoise"]["reveal_strategy"]),
                        },
                    )
                

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
        return {
            "acc_by_iter": epoch_acc_mean,
            "loss_by_iter": epoch_loss_mean,
            "certain_loss_by_iter": epoch_certain_loss_mean,
            "certain_acc_by_iter": epoch_certain_acc_mean,
            "decode_loss_by_iter": epoch_decode_loss_mean,
            "decode_acc_by_iter": epoch_decode_acc_mean,
        }

    # -------------------------
    # Main epoch loop
    # -------------------------
    for epoch in range(start_epoch, num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        # Train
        train_stats = run_epoch(train_loader, train=True, finetune=True, epoch_num=epoch)

        print("Train:")
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
                "decode_mode": str(train_config["denoise"]["reveal_strategy"]),
            },
        )

        # clear cache
        torch.cuda.empty_cache()

        # Save PyTorch checkpoint (talking_ml only + optim/sched)
        save_ckpt(
            args.savedir,
            epoch,
            model,
            optimizer,
            scheduler,
            extra={
                "think_device1": THINK_DEVICE1,
                "think_device2": THINK_DEVICE2,
                "talk_device": TALK_DEVICE,
                "global_step": global_step,
            },
            model_config=model_config
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

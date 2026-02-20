import os, json, math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _extract_lora_state_dict(module: torch.nn.Module) -> dict:
    """
    Return only LoRA parameters from a module's state_dict.
    Heuristic: keys containing 'lora' (covers most LoRA impls).
    You can tighten this if your LoRA library uses different key names.
    """
    sd = module.state_dict()
    lora_sd = {k: v for k, v in sd.items() if "lora" in k.lower()}
    return lora_sd


def _has_lora_params(module: torch.nn.Module) -> bool:
    for n, _ in module.named_parameters():
        if "lora" in n.lower():
            return True
    return False


def save_ckpt(save_root, epoch, model, optimizer, scheduler, extra=None,model_config=None):
    """
    Save:
      - talk_model full state_dict
      - think_model LoRA-only state_dict (if any)
      - optimizer + scheduler
    Assumes `model` has attributes: model.talk_model and model.think_model
    """
    state_dir = os.path.join(save_root, f"state_{epoch}")
    os.makedirs(state_dir, exist_ok=True)
    ckpt_path = os.path.join(state_dir, "ckpt.pt")

    payload = {
        "epoch": epoch,
        "talk_model": model.talk_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    # Save think LoRA if exists
    if hasattr(model, "think_model") and _has_lora_params(model.think_model):
        think_lora = _extract_lora_state_dict(model.think_model)
        if len(think_lora) > 0:
            payload["think_lora"] = think_lora

    if extra:
        payload.update(extra)

    torch.save(payload, ckpt_path)
    if model_config is not None:
        json_path =os.path.join(state_dir, "config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, ensure_ascii=False, indent=2)
    
    print(f"[CKPT] saved: {ckpt_path}  "
          f"(talk={len(payload['talk_model'])} keys, "
          f"think_lora={len(payload.get('think_lora', {}))} keys)")

def load_ckpt(state_dir, model, optimizer=None, scheduler=None, map_location="cpu", strict_talk=True):
    """
    Load:
      - talk_model (strict by default)
      - think_model LoRA-only (if present in ckpt; non-strict load)
      - optimizer/scheduler if provided
    """
    ckpt_path = os.path.join(state_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # 1) talk model
    if "talk_model" not in ckpt:
        raise KeyError(f"ckpt missing 'talk_model': {ckpt_path}")
    model.talk_model.load_state_dict(ckpt["talk_model"], strict=strict_talk)

    # 2) think LoRA (optional)
    if "think_lora" in ckpt:
        if not hasattr(model, "think_model"):
            raise AttributeError("ckpt has think_lora but model has no think_model")
        missing, unexpected = model.think_model.load_state_dict(ckpt["think_lora"], strict=False)
        # strict=False is correct because you are loading a partial state_dict
        if missing:
            print(f"[CKPT] think_lora missing keys (ok if LoRA structure changed): {len(missing)}")
        if unexpected:
            print(f"[CKPT] think_lora unexpected keys: {len(unexpected)}")

    # 3) opt/sched (optional)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt.get("epoch", -1) + 1
    print(f"[CKPT] loaded: {ckpt_path}  -> start_epoch={start_epoch}")
    return start_epoch

@torch.no_grad()
def apply_repetition_penalty_3d(
    logits: torch.Tensor,          # [BG, L, V]
    context_ids: torch.Tensor,     # [BG, L]  (tokens already in the block)
    revealed_mask: torch.Tensor,   # [BG, L]  bool, True where token is "seen"/revealed
    penalty: float = 1.2,
):
    """
    HF-style repetition penalty, applied to logits for tokens that already appeared
    in the revealed part of context_ids (per row).
    """
    if penalty is None or penalty <= 1.0:
        return logits

    BG, L, V = logits.shape
    out = logits.clone()

    for b in range(BG):
        seen = context_ids[b][revealed_mask[b]].tolist()
        if not seen:
            continue
        seen = list(set(seen))  # unique

        # out[b, :, seen] is [L, |seen|]
        sel = out[b, :, seen]

        # HF rule: if logit > 0 -> divide; else multiply
        pos = sel > 0
        sel = torch.where(pos, sel / penalty, sel * penalty)
        out[b, :, seen] = sel

    return out


@torch.no_grad()
def top_p_sample_from_logits_3d(
    logits: torch.Tensor,          # [BG, L, V]
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    """
    Nucleus sampling per position. Returns sampled token ids: [BG, L]
    """
    if temperature is None or temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    if top_p is None or top_p >= 1.0:
        # plain categorical sampling
        probs = F.softmax(logits, dim=-1)
        ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.size(0), logits.size(1))
        return ids

    # sort
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = sorted_probs.cumsum(dim=-1)

    # mask tokens with cum prob > top_p (keep at least 1)
    remove = cum_probs > top_p
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    # sample from filtered distribution
    probs = F.softmax(sorted_logits, dim=-1)
    sample_in_sorted = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.size(0), logits.size(1))

    # map back to original vocab ids
    sampled_ids = sorted_idx.gather(-1, sample_in_sorted.unsqueeze(-1)).squeeze(-1)
    return sampled_ids

def topk_soft_embedding_from_logits(
    logits: torch.Tensor,          # [BG, L, V]
    emb_weight: torch.Tensor,      # [V, D]
    topk: int = 32,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Return soft embeddings: [BG, L, D] computed from top-k logits.
    """
    BG, L, V = logits.shape
    K = min(topk, V)

    vals, idx = logits.topk(K, dim=-1)                 # [BG, L, K]
    probs = torch.softmax(vals / temperature, dim=-1)  # [BG, L, K]

    # gather embedding vectors for those vocab ids
    # emb_k: [BG, L, K, D]
    emb_k = emb_weight.index_select(0, idx.reshape(-1)).reshape(BG, L, K, -1)

    # weighted sum over K
    soft_emb = (probs.unsqueeze(-1) * emb_k).sum(dim=-2)  # [BG, L, D]
    return soft_emb



@torch.no_grad()
def select_reveal_positions(
    loss_mask: torch.Tensor,          # [BG, L] 1=masked(active), 0=already revealed
    logits: Optional[torch.Tensor],    # [BG, L, V] (required for greedy)
    k_reveal: int = 1,
    mode: str = "random",             # "random" | "greedy" | "ar_force"
    generator=None,
) -> torch.Tensor:
    """
    Returns:
      idx: [BG, k] indices to reveal for each row (may include inactive positions if row has <k active)
    """
    device = loss_mask.device
    BG, L = loss_mask.shape
    active = loss_mask.bool()

    k = min(k_reveal, L)
    if k <= 0:
        return torch.empty((BG, 0), dtype=torch.long, device=device)

    if mode == "random":
        if generator is not None:
            scores = torch.rand((BG, L), device=device, generator=generator)
        else:
            scores = torch.rand((BG, L), device=device)
        scores = scores.masked_fill(~active, float("-inf"))
        idx = scores.topk(k=k, dim=1).indices  # [BG, k]
        return idx

    if mode == "greedy":
        if logits is None:
            raise ValueError("mode='greedy' requires logits.")
        # confidence = max prob (or max logit) at each position
        # using max logit is fine because monotonic with softmax
        conf = logits.max(dim=-1).values  # [BG, L]
        conf = conf.masked_fill(~active, float("-inf"))
        idx = conf.topk(k=k, dim=1).indices
        return idx

    if mode == "ar_force":
        # reveal left-to-right masked positions (like AR order inside the block)
        # score = -position_index for active, so topk picks smallest indices
        pos = torch.arange(L, device=device).unsqueeze(0).expand(BG, L)  # [BG, L]
        scores = (-pos).to(torch.float32)
        scores = scores.masked_fill(~active, float("-inf"))
        idx = scores.topk(k=k, dim=1).indices
        return idx

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------
# 2) denoise step with soft embedding + (optional) inference
#    Paper-style: keep some MASK anchor for still-masked positions
# ---------------------------------------------------------
def denoise_k_step_soft_embed_v2(
    input_ids: torch.Tensor,
    target: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    logits: torch.Tensor,
    emb_weight: torch.Tensor,
    k_reveal: int = 1,
    soft_topk: int = 32,
    soft_temp: float = 1.0,
    mode: str = "random",
    generator=None,
    return_pred_ids: bool = False,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    sample_tokens: bool = False,
    mask_token_id: Optional[int] = None,
    lam_max: Optional[float] = None,
    lam_min: Optional[float] = None,
    entropy_temperature: Optional[float] = None,
    detach_lambda: bool = True,
):
    device = input_ids.device
    BG, L = input_ids.shape

    # 0) Optional repetition penalty (inference)
    logits_adj = None
    if repetition_penalty > 1:
        revealed_mask = ~loss_mask.bool()
        logits_adj = apply_repetition_penalty_3d(
            logits=logits,
            context_ids=input_ids,
            revealed_mask=revealed_mask,
            penalty=repetition_penalty,
        )
    logits_used = logits_adj if logits_adj is not None else logits

    # 1) Pick positions to reveal
    idx = select_reveal_positions(
        loss_mask=loss_mask,
        logits=logits_used,
        k_reveal=k_reveal,
        mode=mode,
        generator=generator,
    )  # [BG, k]

    active = loss_mask.bool()
    chosen_active = active.gather(1, idx)  # [BG, k]
    rows = torch.arange(BG, device=device).unsqueeze(1).expand_as(idx)
    rows = rows[chosen_active]
    cols = idx[chosen_active]

    input_ids_next = input_ids.clone()
    loss_mask_next = loss_mask.clone()

    # 2) Choose tokens for revealed positions
    if target is None:
        if sample_tokens:
            pred_ids = top_p_sample_from_logits_3d(
                logits=logits_used, top_p=top_p, temperature=temperature
            )  # [BG, L]
        else:
            pred_ids = logits_used.argmax(dim=-1)  # [BG, L]
        input_ids_next[rows, cols] = pred_ids[rows, cols]
    else:
        pred_ids = logits.argmax(dim=-1)  # for logging only
        input_ids_next[rows, cols] = target[rows, cols]

    loss_mask_next[rows, cols] = 0

    # 3) Build embeddings
    base_emb = F.embedding(input_ids_next, emb_weight).to(dtype=emb_weight.dtype)

    logits_for_soft = logits_used if (target is None and logits_adj is not None) else logits
    soft_emb = topk_soft_embedding_from_logits(
        logits=logits_for_soft,
        emb_weight=emb_weight,
        topk=soft_topk,
        temperature=soft_temp,
    ).to(dtype=emb_weight.dtype)

    m = loss_mask_next.bool().unsqueeze(-1)  # [BG, L, 1]

    # 3.5) Backward-compatible branch
    if (lam_max is None) and (lam_min is None):
        input_emb_next = torch.where(m, soft_emb, base_emb)
    else:
        # 4) Compute λ (entropy gate)
        lam_max = float(0.7 if lam_max is None else lam_max)
        lam_min = float(0.0 if lam_min is None else lam_min)

        ent_temp = soft_temp if entropy_temperature is None else entropy_temperature
        p = F.softmax(logits_for_soft / max(ent_temp, 1e-6), dim=-1)  # [BG, L, V]
        logp = torch.log(p.clamp_min(1e-8))
        H = -(p * logp).sum(dim=-1)  # [BG, L]

        V = logits_for_soft.size(-1)
        Hn = H / math.log(V)  # ~[0,1]
        lam = (1.0 - Hn).clamp(0.0, 1.0)
        lam = (lam_min + (lam_max - lam_min) * lam).to(dtype=emb_weight.dtype)  # [BG, L]
        if detach_lambda:
            lam = lam.detach()

        # 5) Mask anchor
        if mask_token_id is not None:
            mask_emb = emb_weight[mask_token_id].to(dtype=emb_weight.dtype)      # [D]
            mask_emb = mask_emb.view(1, 1, -1).expand(BG, L, -1)                 # [BG, L, D]
        else:
            mask_emb = base_emb  # assume still-masked input_ids are mask ids

        lam3 = lam.unsqueeze(-1)
        mixed = (1.0 - lam3) * mask_emb + lam3 * soft_emb
        input_emb_next = torch.where(m, mixed, base_emb)

    if return_pred_ids:
        return input_ids_next, input_emb_next, loss_mask_next, pred_ids
    return input_ids_next, input_emb_next, loss_mask_next
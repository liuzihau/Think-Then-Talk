import os, json, math
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

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
    if hasattr(model, "talk_lm_head_weight") and model.talk_lm_head_weight is not None:
        payload["talk_lm_head_weight"] = model.talk_lm_head_weight.detach().cpu()
    if hasattr(model, "talk_lm_head_bias") and model.talk_lm_head_bias is not None:
        payload["talk_lm_head_bias"] = model.talk_lm_head_bias.detach().cpu()

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

    # optional: talk lm_head params (newer checkpoints)
    if "talk_lm_head_weight" in ckpt and hasattr(model, "talk_lm_head_weight") and model.talk_lm_head_weight is not None:
        w = ckpt["talk_lm_head_weight"].to(model.talk_lm_head_weight.device, dtype=model.talk_lm_head_weight.dtype)
        if isinstance(model.talk_lm_head_weight, torch.nn.Parameter):
            model.talk_lm_head_weight.data.copy_(w)
        else:
            model.talk_lm_head_weight = w
    if "talk_lm_head_bias" in ckpt and hasattr(model, "talk_lm_head_bias") and model.talk_lm_head_bias is not None:
        b = ckpt["talk_lm_head_bias"].to(model.talk_lm_head_bias.device, dtype=model.talk_lm_head_bias.dtype)
        if isinstance(model.talk_lm_head_bias, torch.nn.Parameter):
            model.talk_lm_head_bias.data.copy_(b)
        else:
            model.talk_lm_head_bias = b

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


def _sample_policy_mode(
    policy: Any,
    *,
    device: torch.device,
    generator=None,
    default_mode: str,
) -> str:
    if policy is None:
        return default_mode
    if isinstance(policy, str):
        return policy
    if not isinstance(policy, dict):
        raise ValueError(f"Unsupported policy type: {type(policy)}")

    policy_type = str(policy.get("type", "fixed"))
    if policy_type in {"fixed", "single"}:
        mode = policy.get("value", policy.get("mode", policy.get("name", default_mode)))
        return str(mode)
    if policy_type != "mixture":
        raise ValueError(f"Unsupported policy type: {policy_type}")

    choices = policy.get("choices", {})
    if not isinstance(choices, dict) or not choices:
        raise ValueError("Mixture policy requires a non-empty 'choices' dict.")

    names = []
    weights = []
    for name, weight in choices.items():
        w = float(weight)
        if w < 0:
            raise ValueError(f"Mixture weight must be non-negative, got {w} for {name}")
        if w == 0:
            continue
        names.append(str(name))
        weights.append(w)

    if not names:
        raise ValueError("Mixture policy has no positive-probability choices.")

    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    choice_idx = torch.multinomial(weights_t, num_samples=1, generator=generator).item()
    return names[choice_idx]


def get_policy_label(policy: Any, default_mode: str) -> str:
    if policy is None:
        return default_mode
    if isinstance(policy, str):
        return policy
    if isinstance(policy, dict):
        policy_type = str(policy.get("type", "fixed"))
        if policy_type in {"fixed", "single"}:
            return str(policy.get("value", policy.get("mode", policy.get("name", default_mode))))
        if policy_type == "mixture":
            choices = policy.get("choices", {})
            if isinstance(choices, dict) and choices:
                return max(choices.items(), key=lambda kv: float(kv[1]))[0]
    return default_mode


def _normalize_decode_mode_name(mode: Any) -> Any:
    if not isinstance(mode, str):
        return mode
    if mode in {"greedy_threshold", "confidence_threshold"}:
        return "greedy"
    return mode


def _normalize_decode_policy(policy: Any) -> Any:
    if isinstance(policy, str):
        return _normalize_decode_mode_name(policy)
    if not isinstance(policy, dict):
        return policy

    policy = dict(policy)
    policy_type = str(policy.get("type", "fixed"))
    if policy_type == "mixture":
        raw_choices = policy.get("choices", {})
        if isinstance(raw_choices, dict):
            merged_choices = {}
            for name, weight in raw_choices.items():
                norm_name = _normalize_decode_mode_name(name)
                merged_choices[norm_name] = merged_choices.get(norm_name, 0.0) + float(weight)
            policy["choices"] = merged_choices
        return policy

    for key in ("value", "mode", "name"):
        if key in policy:
            policy[key] = _normalize_decode_mode_name(policy[key])
    return policy


def get_denoise_reveal_config(denoise_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    denoise_cfg = denoise_cfg or {}
    reveal = denoise_cfg.get("reveal", {})
    reveal = dict(reveal) if isinstance(reveal, dict) else {}

    reveal["k"] = int(reveal.get("k", denoise_cfg.get("reveal_k", 1)))
    reveal["policy"] = reveal.get("policy", denoise_cfg.get("reveal_strategy", "random"))
    return reveal


def get_denoise_decode_config(
    denoise_cfg: Optional[Dict[str, Any]],
    *,
    default_k: int,
) -> Dict[str, Any]:
    denoise_cfg = denoise_cfg or {}
    decode = denoise_cfg.get("decode", {})
    decode = dict(decode) if isinstance(decode, dict) else {}

    decode["policy"] = _normalize_decode_policy(decode.get("policy", "fix"))
    decode["fix_k"] = int(decode.get("fix_k", default_k))
    decode["max_k"] = int(decode.get("max_k", decode["fix_k"]))
    decode["min_k"] = int(decode.get("min_k", 1))
    decode["confidence_threshold"] = float(
        decode.get("confidence_threshold", decode.get("threshold", 0.0))
    )
    return decode


def _build_reveal_scores(
    loss_mask: torch.Tensor,
    logits: Optional[torch.Tensor],
    *,
    mode: str,
    generator=None,
) -> torch.Tensor:
    device = loss_mask.device
    BG, L = loss_mask.shape
    active = loss_mask.bool()

    if mode == "random":
        if generator is not None:
            scores = torch.rand((BG, L), device=device, generator=generator)
        else:
            scores = torch.rand((BG, L), device=device)
    elif mode == "greedy":
        if logits is None:
            raise ValueError("mode='greedy' requires logits.")
        scores = logits.max(dim=-1).values
    elif mode == "ar_force":
        pos = torch.arange(L, device=device).unsqueeze(0).expand(BG, L)
        scores = (-pos).to(torch.float32)
    else:
        raise ValueError(f"Unknown reveal mode: {mode}")

    return scores.masked_fill(~active, float("-inf"))


def resolve_denoise_positions(
    loss_mask: torch.Tensor,
    logits: Optional[torch.Tensor],
    *,
    reveal_cfg: Optional[Dict[str, Any]] = None,
    decode_cfg: Optional[Dict[str, Any]] = None,
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    device = loss_mask.device
    BG, L = loss_mask.shape
    active = loss_mask.bool()
    active_counts = active.sum(dim=1)

    reveal_cfg = dict(reveal_cfg or {})
    decode_cfg = dict(decode_cfg or {})

    reveal_k = int(reveal_cfg.get("k", 1))
    decode_fix_k = int(decode_cfg.get("fix_k", reveal_k))
    decode_max_k = int(decode_cfg.get("max_k", decode_fix_k))
    decode_min_k = int(decode_cfg.get("min_k", 1))
    candidate_k = max(1, min(L, max(reveal_k, decode_fix_k, decode_max_k)))

    reveal_mode = _sample_policy_mode(
        reveal_cfg.get("policy"),
        device=device,
        generator=generator,
        default_mode="random",
    )
    decode_mode = _sample_policy_mode(
        decode_cfg.get("policy"),
        device=device,
        generator=generator,
        default_mode="fix",
    )

    scores = _build_reveal_scores(
        loss_mask=loss_mask,
        logits=logits,
        mode=reveal_mode,
        generator=generator,
    )
    idx = scores.topk(k=candidate_k, dim=1).indices
    chosen_active = active.gather(1, idx)

    if decode_mode == "fix":
        reveal_counts = torch.full((BG,), decode_fix_k, dtype=torch.long, device=device)
    elif decode_mode == "greedy":
        if logits is None:
            raise ValueError(f"decode mode '{decode_mode}' requires logits.")
        conf = F.softmax(logits, dim=-1).amax(dim=-1)
        threshold = float(decode_cfg.get("confidence_threshold", 0.0))
        above = (conf > threshold) & active
        reveal_counts = above.sum(dim=1).to(torch.long)
        reveal_counts = reveal_counts.clamp(max=decode_max_k)
        reveal_counts = torch.where(
            active_counts > 0,
            torch.maximum(reveal_counts, torch.full_like(reveal_counts, decode_min_k)),
            reveal_counts,
        )
    else:
        raise ValueError(f"Unknown decode mode: {decode_mode}")

    reveal_counts = torch.minimum(reveal_counts, active_counts.to(torch.long))
    rank = torch.arange(candidate_k, device=device).unsqueeze(0).expand(BG, candidate_k)
    chosen_active = chosen_active & (rank < reveal_counts.unsqueeze(1))
    return idx, chosen_active, reveal_mode, decode_mode

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
    mode: Any = "random",             # "random" | "greedy" | "ar_force" | mixture policy
    decode_cfg: Optional[Dict[str, Any]] = None,
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    Returns:
      idx: [BG, k] candidate indices to reveal for each row
      chosen_active: [BG, k] bool mask indicating which candidates are actually selected
      reveal_mode: sampled reveal policy used for this step
      decode_mode: sampled decode policy used for this step
    """
    device = loss_mask.device
    BG, _ = loss_mask.shape
    k = max(int(k_reveal), 0)
    if k <= 0:
        empty = torch.empty((BG, 0), dtype=torch.long, device=device)
        empty_mask = torch.zeros((BG, 0), dtype=torch.bool, device=device)
        return empty, empty_mask, get_policy_label(mode, "random"), get_policy_label(
            None if decode_cfg is None else decode_cfg.get("policy"),
            "fix",
        )

    reveal_cfg = {"k": k, "policy": mode}
    decode_cfg = dict(decode_cfg or {})
    decode_cfg["fix_k"] = int(decode_cfg.get("fix_k", k))
    decode_cfg["max_k"] = int(decode_cfg.get("max_k", decode_cfg["fix_k"]))

    return resolve_denoise_positions(
        loss_mask=loss_mask,
        logits=logits,
        reveal_cfg=reveal_cfg,
        decode_cfg=decode_cfg,
        generator=generator,
    )


def denoise_k_step_hard(
    input_ids: torch.Tensor,
    target: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    logits: torch.Tensor,
    *,
    reveal_cfg: Dict[str, Any],
    decode_cfg: Dict[str, Any],
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    device = input_ids.device
    BG = input_ids.size(0)

    idx, chosen_active, reveal_mode, decode_mode = resolve_denoise_positions(
        loss_mask=loss_mask,
        logits=logits,
        reveal_cfg=reveal_cfg,
        decode_cfg=decode_cfg,
        generator=generator,
    )

    rows = torch.arange(BG, device=device).unsqueeze(1).expand_as(idx)
    rows = rows[chosen_active]
    cols = idx[chosen_active]

    input_ids_next = input_ids.clone()
    loss_mask_next = loss_mask.clone()
    fill_ids = logits.argmax(dim=-1) if target is None else target

    if rows.numel() > 0:
        input_ids_next[rows, cols] = fill_ids[rows, cols]
        loss_mask_next[rows, cols] = 0

    return input_ids_next, loss_mask_next, reveal_mode, decode_mode


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
    mode: Any = "random",
    decode_cfg: Optional[Dict[str, Any]] = None,
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
    idx, chosen_active, _, _ = select_reveal_positions(
        loss_mask=loss_mask,
        logits=logits_used,
        k_reveal=k_reveal,
        mode=mode,
        decode_cfg=decode_cfg,
        generator=generator,
    )
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

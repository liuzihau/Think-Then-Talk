import os
import torch


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


def save_ckpt(save_root, epoch, model, optimizer, scheduler, extra=None):
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
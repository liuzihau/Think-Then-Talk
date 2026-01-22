import os
import torch


def _safe_tok(tokenizer, tid: int) -> str:
    s = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    if s == " ":
        s = "␠"
    return s


def _fmt_token_list(tokenizer, ids, max_items=80):
    ids = list(map(int, ids))
    shown = ids[:max_items]
    tail = "" if len(ids) <= max_items else f" ... (+{len(ids)-max_items})"
    return " ".join([f"{_safe_tok(tokenizer, t)}({t})" for t in shown]) + tail


def _fmt_pairs(tokenizer, tgt_ids, pred_ids, mask01, max_items=64):
    # mask01: 1 => still need predict (show pred), 0 => revealed (show gt)
    out = []
    L = len(tgt_ids)
    for i in range(L):
        tgt = int(tgt_ids[i])
        pred = int(pred_ids[i])
        if int(mask01[i]) == 0:
            out.append(f"[{_safe_tok(tokenizer,tgt)}({tgt}) , gt]")
        else:
            out.append(f"[{_safe_tok(tokenizer,tgt)}({tgt}) , {_safe_tok(tokenizer,pred)}({pred})]")
    if L > max_items:
        out = out[:max_items] + [f"... (+{L-max_items})"]
    return " ".join(out)


@torch.no_grad()
def visualize_t3_batch_trace(
    *,
    tokenizer,
    data,                 # your collator output (on CPU/GPU)
    block_size: int,
    block_num: int,
    bidx: int,            # which batch sample to print
    target_BG_L,          # [B*G, L] on any device
    input_ids_init_BG_L,  # [B*G, L] initial talk input_ids (masked) before denoise
    steps_pred_BG_L,      # list of [B*G, L] argmax pred per denoise step
    steps_mask_BG_L,      # list of [B*G, L] loss_mask per denoise step (BEFORE denoise update)
    max_ctx_tokens: int = 120,
    max_pairs_per_block: int = 64,
    max_steps: int = 6,
    tag: str = "",
    log_path: str = "t3_trace.log",   # <-- NEW
):
    """
    Appends trace info to `log_path` instead of printing to stdout.
    """

    def _w(line=""):
        f.write(line + "\n")

    # open in append mode
    parent = os.path.dirname(log_path)
    if parent:  # empty if path has no directory component
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:

        # ----- reconstruct seq boundaries from collator output -----
        attn = data["attention_mask"][bidx].detach().to("cpu")
        seq_len = int(attn.sum().item())

        lm = data["loss_mask"][bidx].detach().to("cpu")
        mask_len = int(lm.sum().item())
        mask_start = seq_len - mask_len

        seq = data["input_ids"][bidx, :seq_len].detach().to("cpu")

        gt_ctx_len = max(0, mask_len - block_size)
        gt_ctx_start = max(0, mask_start - gt_ctx_len)

        context_ids = seq[:mask_start]
        gt_ctx_ids = seq[gt_ctx_start:mask_start]
        mask_ids = seq[mask_start:seq_len]

        _w("")
        _w("=" * 120)
        _w(f"[VIS]{(' ' + tag) if tag else ''}  bidx={bidx}  seq_len={seq_len}  mask_len={mask_len}  G={block_num}  L={block_size}")
        _w("Context (inp | prefix | gt_context):")

        ctx_show = context_ids.tolist()
        if len(ctx_show) > max_ctx_tokens:
            ctx_show = ctx_show[-max_ctx_tokens:]
            _w(f"  ... (showing last {max_ctx_tokens} tokens)")

        _w("  " + _fmt_token_list(tokenizer, ctx_show, max_items=max_ctx_tokens))
        _w("GT context tail (window[:-block_size]) right before masks:")
        _w("  " + _fmt_token_list(tokenizer, gt_ctx_ids.tolist(), max_items=max_ctx_tokens))
        _w("Masks (initial talk input span):")
        _w("  " + _fmt_token_list(tokenizer, mask_ids.tolist(), max_items=max_ctx_tokens))

        # ----- reshape talk tensors back to [B, G, L] -----
        B = data["input_ids"].size(0)
        G = block_num
        L = block_size
        assert target_BG_L.size(0) == B * G and target_BG_L.size(1) == L

        tgt = target_BG_L.detach().to("cpu").view(B, G, L)[bidx]
        x0  = input_ids_init_BG_L.detach().to("cpu").view(B, G, L)[bidx]

        _w("Step -1 (initial masked input_ids per block):")
        for g in range(G):
            _w(f"  Block {g:02d}: " + _fmt_token_list(tokenizer, x0[g].tolist(), max_items=max_pairs_per_block))

        S = min(len(steps_pred_BG_L), max_steps)
        for s in range(S):
            pred = steps_pred_BG_L[s].detach().to("cpu").view(B, G, L)[bidx]
            msk  = steps_mask_BG_L[s].detach().to("cpu").view(B, G, L)[bidx]

            _w("-" * 120)
            _w(f"Step {s}:")
            for g in range(G):
                line = _fmt_pairs(
                    tokenizer,
                    tgt[g].tolist(),
                    pred[g].tolist(),
                    msk[g].tolist(),
                    max_items=max_pairs_per_block,
                )
                _w(f"  Block {g:02d} {line}")

        _w("=" * 120)
        _w("")

        f.flush()


@torch.no_grad()
def visualize_attention_bias_blocks(
    *,
    tokenizer,
    data,
    bidx: int,
    block_size: int,
    block_num: int,
    show_last: int = 128,
    query_mode: str = "last",   # "last" or "first_last"
    max_ctx_tokens: int = 80,
    log_path: str = "t3_attn.log",   # <-- NEW
):
    """
    Appends attention-bias visibility info to `log_path` instead of printing.
    """

    def _w(line=""):
        f.write(line + "\n")

    with open(log_path, "a", encoding="utf-8") as f:

        attn = data["attention_mask"][bidx].detach().to("cpu")
        seq_len = int(attn.sum().item())

        seq = data["input_ids"][bidx, :seq_len].detach().to("cpu")

        lm = data["loss_mask"][bidx, :seq_len].detach().to("cpu")
        mask_len = int(lm.sum().item())
        mask_start = seq_len - mask_len

        bias = data["attention_bias"][bidx, 0, :seq_len, :seq_len].detach().to("cpu")

        ctx_tail = seq[max(0, mask_start - max_ctx_tokens):mask_start].tolist()

        _w("")
        _w("#" * 120)
        _w(f"[ATTN VIS] bidx={bidx} seq_len={seq_len} mask_len={mask_len} mask_start={mask_start}")
        _w("Context tail right before masks:")
        _w("  " + _fmt_token_list(tokenizer, ctx_tail, max_items=max_ctx_tokens))
        _w("#" * 120)

        def _print_visible_for_query(q: int, label: str):
            row = bias[q]
            allowed = (row == 0)

            idxs = torch.nonzero(allowed, as_tuple=False).squeeze(1).tolist()
            if len(idxs) == 0:
                _w(f"    {label}: (no visible tokens)")
                return

            tail = idxs[-show_last:]
            toks = []
            for k in tail:
                tid = int(seq[k].item())
                toks.append(f"{k:04d}:{_safe_tok(tokenizer, tid)}({tid})")

            _w(f"    {label}: visible={len(idxs)}  (show last {min(show_last, len(tail))})")
            _w("      " + " | ".join(toks))

        for g in range(block_num):
            q0 = mask_start + g * block_size
            q1 = mask_start + (g + 1) * block_size - 1
            if q0 >= seq_len:
                break
            q1 = min(q1, seq_len - 1)

            _w("")
            _w(f"Block {g:02d} query_range=[{q0}, {q1}]")

            if query_mode == "last":
                _print_visible_for_query(q1, label=f"q={q1} (last token in block)")
            elif query_mode == "first_last":
                _print_visible_for_query(q0, label=f"q={q0} (first token in block)")
                if q1 != q0:
                    _print_visible_for_query(q1, label=f"q={q1} (last token in block)")
            else:
                raise ValueError(f"Unknown query_mode={query_mode}")

        _w("")
        f.flush()

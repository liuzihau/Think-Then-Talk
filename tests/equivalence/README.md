# Axis 3 — Inference equivalence runbook

This directory holds the manual-gate scripts that pin down what the axis 3
refactor changed and what it didn't, by comparing token-level inference
output across commits.

## Why two captures (inline + engine)?

Axis 3 split into two commits with different blast radius:

| Commit | What changed | Math change? |
|--------|--------------|:-----------:|
| `4e7dfc5` ("extract T3DecodeEngine + DenoiseOverrides…") | Pure refactor; new code paths added but default behaviour preserved. | No |
| `eb4ad38` ("prefer_no_mask fast path…") | New `prefer_no_mask` kwarg routes the inference think pass through FlashAttention instead of SDPA-with-zero-bias. | Last-bit drift possible. |

To validate them separately we capture the same prompt three times:

1. **`baseline.json`** — at the pre-axis-3 ref (default `HEAD~2`), running an
   inline reproduction of the original per-block decode loop. This is the
   gold reference.
2. **`refactor_only.json`** — at the current ref, running the **same inline
   loop**. This deliberately bypasses `T3DecodeEngine` so it isolates the
   pure-refactor delta. **Must be byte-identical to baseline** — anything
   else is a key-drift / signature-rename bug.
3. **`engine.json`** — at the current ref, running through `T3DecodeEngine`
   (which uses `prefer_no_mask=True`). May drift from baseline at the last
   bit due to FlashAttention vs SDPA reduction order. Token drift is OK iff
   answer-level GSM8K accuracy holds; that's stage 4 of the pipeline.

## Usage

End-to-end orchestration is in `scripts/run_axis3_validation.sh`.

```bash
T3_CKPT_PATH=/path/to/state_2 bash scripts/run_axis3_validation.sh
```

For just the inference equivalence stages (no GSM8K), this is the default.
Run with `T3_RUN_GSM8K=1` to add the heavier accuracy gate at the end.

To capture a single point manually:

```bash
python tests/equivalence/capture_inference.py \
    --ckpt_path /path/to/state_2 \
    --prompt "Solve: 17 + 35 = ?" \
    --gen_length 48 --block_size 6 --seed 0 \
    --mode inline \
    --output /tmp/baseline.json
```

Then diff:

```bash
python tests/equivalence/diff_tokens.py /tmp/baseline.json /tmp/current.json
```

`diff_tokens.py` exits 0 on full match, 1 on any mismatch, so the orchestrator
can gate on `$?`. `--max_diff N` allows up to N generated-token mismatches
(used for the engine-vs-baseline comparison).

## What "inline mode" actually runs

`capture_inference.py --mode inline` reimplements the pre-axis-3 per-block
decode loop directly against `T3Model`:

- builds the dense block-causal `attention_bias` up front,
- per block: think forward with bias → trim trailing block from KV cache
  → select talk inputs → denoise loop (hard or soft per `model_config`).

It deliberately does **not** import `model.inference_engine.T3DecodeEngine`,
so the script runs unchanged on commits before the engine existed (i.e.
on the pre-axis-3 ref). That's how baseline and refactor-only captures are
produced from the same script under two different `git checkout`s.

## Common failure modes

- **Tokenizer differs at prompt level.** `diff_tokens.py` reports
  `prompt_token_ids differ at index N`. Almost always means the
  `pretrained_model_name_or_path` in `config.json` resolved to different
  HF revisions, or the chat template changed. Pin the HF cache and
  re-run.
- **Generated tokens diverge from byte 0.** `T3Model.forward` semantics
  changed somewhere — check the most recent diff against `model/modeling_t3.py`
  and `model/LLaDA/modeling_llada.py`. Likely a default-value change in a
  newly-added kwarg.
- **Generated tokens diverge after a few positions.** Numerical drift
  accumulating through the denoise loop. Expected on the engine-vs-baseline
  diff (FlashAttention path); not expected on the inline-vs-inline diff.
- **Engine diff exceeds `--max_diff` but answer is still correct.** Bump
  `T3_MAX_ENGINE_DIFF` in the orchestrator after confirming via stage 4
  GSM8K that accuracy is preserved. Document the bump in
  `T3_h200_refactor/validation/axis3_metrics.md`.

#!/usr/bin/env bash
#
# scripts/run_axis3_validation.sh — drives the full axis 3 validation pipeline.
#
# Stages:
#   1. Pytest gates (checkpoint compat + inference smoke).
#   2. Inference equivalence: capture baseline (HEAD~2) and current (HEAD)
#      under the inline-decode mode; require byte-identical tokens.
#   3. Engine vs baseline drift: capture HEAD via T3DecodeEngine; compare to
#      baseline. Token drift is allowed iff GSM8K accuracy is preserved (run
#      stage 4 to verify).
#   4. (Optional) GSM8K eval limit-50 on baseline and current; diff scores.
#
# Required env:
#   T3_CKPT_PATH        Path to a saved T3 checkpoint dir (contains config.json,
#                       ckpt.pt). Used by every stage.
#
# Optional env:
#   T3_BASELINE_REF     Git ref to use as the pre-axis-3 baseline.
#                       Default: HEAD~2 (one commit before the axis 3 pair).
#   T3_PROMPT           Inference prompt. Default: "Solve: 17 + 35 = ?"
#   T3_GEN_LENGTH       Default 48.
#   T3_BLOCK_SIZE       Default 6.
#   T3_SEED             Default 0.
#   T3_OUT_DIR          Where capture JSONs land. Default: ./validation_axis3.
#   T3_RUN_GSM8K        "1" to run stage 4. Default: "0" (skip — heavier).
#   T3_MAX_ENGINE_DIFF  Allowed generated-token mismatches between baseline and
#                       engine capture. Default 0; bump if FlashAttention vs
#                       SDPA introduces last-bit drift you've decided to accept.
#
# Run from the Think-Then-Talk repo root:
#   T3_CKPT_PATH=/path/to/state_2 bash scripts/run_axis3_validation.sh
#
# The script aborts on the first failed gate. Captures land in T3_OUT_DIR for
# manual inspection regardless of pass/fail.
#
set -euo pipefail

: "${T3_CKPT_PATH:?T3_CKPT_PATH must be set to a checkpoint directory}"
# Pinned to the explicit pre-axis-3 commit so this default doesn't rot as
# more commits land on top of the axis 3 pair.
T3_BASELINE_REF=${T3_BASELINE_REF:-f25cc1e}
T3_PROMPT=${T3_PROMPT:-"Solve: 17 + 35 = ?"}
T3_GEN_LENGTH=${T3_GEN_LENGTH:-48}
T3_BLOCK_SIZE=${T3_BLOCK_SIZE:-6}
T3_SEED=${T3_SEED:-0}
T3_OUT_DIR=${T3_OUT_DIR:-./validation_axis3}
T3_RUN_GSM8K=${T3_RUN_GSM8K:-0}
T3_MAX_ENGINE_DIFF=${T3_MAX_ENGINE_DIFF:-0}

mkdir -p "${T3_OUT_DIR}"
echo "[axis3] outputs -> ${T3_OUT_DIR}"
echo "[axis3] baseline ref: ${T3_BASELINE_REF}"
echo "[axis3] checkpoint  : ${T3_CKPT_PATH}"

# The capture script is copied to /tmp so it survives the git checkout, but
# Python sets sys.path[0] to the script's directory (/tmp), not the CWD —
# so `import model` fails without PYTHONPATH pointing at the repo root.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CURRENT_REF=$(git rev-parse --abbrev-ref HEAD)
if [[ "${CURRENT_REF}" == "HEAD" ]]; then
  CURRENT_REF=$(git rev-parse HEAD)
fi
echo "[axis3] current ref : ${CURRENT_REF}"

if [[ -n $(git status --porcelain) ]]; then
  echo "[axis3] FATAL: working tree is dirty. Commit or stash before running." >&2
  exit 2
fi

cleanup() {
  if [[ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)" != "${CURRENT_REF}" ]]; then
    echo "[axis3] restoring branch ${CURRENT_REF}"
    git checkout -q "${CURRENT_REF}" || true
  fi
}
trap cleanup EXIT

# --- Stage 1: pytest gates ----------------------------------------------------
echo
echo "=========================================================================="
echo "[axis3] stage 1 — pytest gates (checkpoint compat + inference smoke)"
echo "=========================================================================="
T3_CKPT_PATH="${T3_CKPT_PATH}" pytest tests/test_checkpoint_compat.py tests/test_inference_smoke.py -v

# Stash the capture script so it travels with us through the git checkout.
CAPTURE_SCRIPT_TMP=$(mktemp --suffix=.py)
cp tests/equivalence/capture_inference.py "${CAPTURE_SCRIPT_TMP}"
DIFF_SCRIPT_TMP=$(mktemp --suffix=.py)
cp tests/equivalence/diff_tokens.py "${DIFF_SCRIPT_TMP}"

# --- Stage 2: baseline capture (HEAD~2) ---------------------------------------
echo
echo "=========================================================================="
echo "[axis3] stage 2 — capture baseline at ${T3_BASELINE_REF} (inline decode)"
echo "=========================================================================="
git checkout -q "${T3_BASELINE_REF}"
python "${CAPTURE_SCRIPT_TMP}" \
  --ckpt_path "${T3_CKPT_PATH}" \
  --prompt "${T3_PROMPT}" \
  --gen_length "${T3_GEN_LENGTH}" \
  --block_size "${T3_BLOCK_SIZE}" \
  --seed "${T3_SEED}" \
  --mode inline \
  --output "${T3_OUT_DIR}/baseline.json"

# --- Stage 2b: current capture (HEAD, inline) ---------------------------------
echo
echo "=========================================================================="
echo "[axis3] stage 2b — capture current at ${CURRENT_REF} (inline decode)"
echo "=========================================================================="
git checkout -q "${CURRENT_REF}"
python "${CAPTURE_SCRIPT_TMP}" \
  --ckpt_path "${T3_CKPT_PATH}" \
  --prompt "${T3_PROMPT}" \
  --gen_length "${T3_GEN_LENGTH}" \
  --block_size "${T3_BLOCK_SIZE}" \
  --seed "${T3_SEED}" \
  --mode inline \
  --output "${T3_OUT_DIR}/refactor_only.json"

echo
echo "[axis3] gate B — baseline vs refactor (must be byte-identical):"
python "${DIFF_SCRIPT_TMP}" \
  "${T3_OUT_DIR}/baseline.json" \
  "${T3_OUT_DIR}/refactor_only.json" \
  --max_diff 0

# --- Stage 3: engine capture (HEAD, T3DecodeEngine) ---------------------------
echo
echo "=========================================================================="
echo "[axis3] stage 3 — capture current via T3DecodeEngine (engine + prefer_no_mask)"
echo "=========================================================================="
python "${CAPTURE_SCRIPT_TMP}" \
  --ckpt_path "${T3_CKPT_PATH}" \
  --prompt "${T3_PROMPT}" \
  --gen_length "${T3_GEN_LENGTH}" \
  --block_size "${T3_BLOCK_SIZE}" \
  --seed "${T3_SEED}" \
  --mode engine \
  --output "${T3_OUT_DIR}/engine.json"

echo
echo "[axis3] gate C — baseline vs engine (drift up to T3_MAX_ENGINE_DIFF=${T3_MAX_ENGINE_DIFF} tokens):"
python "${DIFF_SCRIPT_TMP}" \
  "${T3_OUT_DIR}/baseline.json" \
  "${T3_OUT_DIR}/engine.json" \
  --max_diff "${T3_MAX_ENGINE_DIFF}"

# --- Stage 4: optional GSM8K limit-50 -----------------------------------------
if [[ "${T3_RUN_GSM8K}" == "1" ]]; then
  echo
  echo "=========================================================================="
  echo "[axis3] stage 4 — GSM8K limit-50 on baseline and current"
  echo "=========================================================================="
  echo "[axis3] checking out ${T3_BASELINE_REF} for baseline GSM8K..."
  git checkout -q "${T3_BASELINE_REF}"
  CKPT_PATH="${T3_CKPT_PATH}" bash eval_t3.sh gsm8k --limit 50 \
    2>&1 | tee "${T3_OUT_DIR}/gsm8k_baseline.log"

  echo "[axis3] checking out ${CURRENT_REF} for current GSM8K..."
  git checkout -q "${CURRENT_REF}"
  CKPT_PATH="${T3_CKPT_PATH}" bash eval_t3.sh gsm8k --limit 50 \
    2>&1 | tee "${T3_OUT_DIR}/gsm8k_current.log"

  echo "[axis3] compare exact_match scores in:"
  echo "         ${T3_OUT_DIR}/gsm8k_baseline.log"
  echo "         ${T3_OUT_DIR}/gsm8k_current.log"
  echo "         (lm_eval prints a summary table near the bottom of each log.)"
else
  echo "[axis3] stage 4 skipped (T3_RUN_GSM8K=0). Run manually with T3_RUN_GSM8K=1 to gate accuracy."
fi

echo
echo "[axis3] all configured stages passed."

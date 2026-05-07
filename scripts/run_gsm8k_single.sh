#!/usr/bin/env bash
#
# scripts/run_gsm8k_single.sh — run GSM8K once on the currently checked-out
# ref, park the output under T3_OUT_DIR with a T3_LABEL suffix, and print
# the exact_match row.
#
# Use when you've already captured baseline / current and only need to add
# one more configuration (e.g. an option-A re-run after fixing prefer_no_mask
# pollution). Does not touch git refs — runs whatever HEAD currently is.
#
# Required env:
#   T3_CKPT_PATH        Checkpoint dir (contains config.json, ckpt.pt).
#
# Optional env:
#   T3_LABEL            Suffix for the parked output dir / log. Default "current".
#   T3_LIMIT            lm_eval --limit. Default 50. "" / "none" / "0" → full set.
#   T3_OUT_DIR          Default ./validation_axis3.
#   T3_GSM8K_GEN_LENGTH Default 512.
#   T3_BLOCK_SIZE       Default 6.
#   CUDA_NO             Forwarded to eval_t3.sh as the cuda index. Default 3.
#
# Usage from the Think-Then-Talk repo root:
#   T3_LABEL=optionA T3_LIMIT="" CUDA_NO=0 \
#     T3_CKPT_PATH=cfg6-...-state_2 \
#     bash scripts/run_gsm8k_single.sh
#
set -euo pipefail

: "${T3_CKPT_PATH:?T3_CKPT_PATH must be set to a checkpoint directory}"
T3_LABEL=${T3_LABEL:-current}
T3_LIMIT=${T3_LIMIT-50}
T3_OUT_DIR=${T3_OUT_DIR:-./validation_axis3}
T3_GSM8K_GEN_LENGTH=${T3_GSM8K_GEN_LENGTH:-512}
T3_BLOCK_SIZE=${T3_BLOCK_SIZE:-6}

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
mkdir -p "${T3_OUT_DIR}"

case "${T3_LIMIT,,}" in
  ""|"none"|"0")
    LIMIT_ARGS=()
    LIMIT_LABEL="<full dataset>"
    ;;
  *)
    LIMIT_ARGS=(--limit "${T3_LIMIT}")
    LIMIT_LABEL="${T3_LIMIT}"
    ;;
esac

EVAL_OUT_ROOT="evals_results/gsm8k-${T3_GSM8K_GEN_LENGTH}"
LOG_PATH="${T3_OUT_DIR}/gsm8k_${T3_LABEL}.log"
PARK_DIR="${T3_OUT_DIR}/gsm8k_${T3_LABEL}_evals_results"

CURRENT_REF=$(git rev-parse --short HEAD)
echo "[run_gsm8k_single] label=${T3_LABEL}  ref=${CURRENT_REF}  limit=${LIMIT_LABEL}"
echo "[run_gsm8k_single] ckpt=${T3_CKPT_PATH}"
echo "[run_gsm8k_single] log -> ${LOG_PATH}"

rm -rf "${EVAL_OUT_ROOT}"

set +e
CKPT_PATH="${T3_CKPT_PATH}" \
GSM8K_GEN_LENGTH="${T3_GSM8K_GEN_LENGTH}" \
BLOCK_SIZE="${T3_BLOCK_SIZE}" \
  bash eval_t3.sh gsm8k "${LIMIT_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
rc=${PIPESTATUS[0]}
set -e
if [[ ${rc} -ne 0 ]]; then
  echo "[run_gsm8k_single] FAILED with exit ${rc}; see ${LOG_PATH}" >&2
  exit ${rc}
fi

rm -rf "${PARK_DIR}"
if [[ -d "${EVAL_OUT_ROOT}" ]]; then
  mv "${EVAL_OUT_ROOT}" "${PARK_DIR}"
  echo "[run_gsm8k_single] parked output -> ${PARK_DIR}"
fi

echo
echo "=========================================================================="
echo "[run_gsm8k_single] exact_match for ${T3_LABEL} (${CURRENT_REF})"
echo "=========================================================================="
grep -E '\|gsm8k\|.*exact_match' "${LOG_PATH}" || echo "(no exact_match row found)"
echo
echo "Compare with existing logs in ${T3_OUT_DIR}/:"
for f in "${T3_OUT_DIR}"/gsm8k_*.log; do
  [[ -f "${f}" ]] || continue
  label=$(basename "${f}" .log)
  label=${label#gsm8k_}
  echo "-- ${label} --"
  grep -E '\|gsm8k\|.*exact_match' "${f}" || echo "  (no exact_match row)"
done

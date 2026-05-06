#!/usr/bin/env bash
#
# scripts/compare_gsm8k.sh — Axis 3 GSM8K accuracy gate (limit-50).
#
# Runs `eval_t3.sh gsm8k --limit ${T3_LIMIT}` at the pre-axis-3 baseline
# (HEAD~2 by default) and at the current ref, parks each run's output dir
# under T3_OUT_DIR with a ref suffix, then prints the exact_match score
# from each run's log so you can eyeball the diff.
#
# This is the standalone version of stage 4 in run_axis3_validation.sh —
# use it when you only want the accuracy gate (e.g. after gates 1-3 already
# passed in a separate run).
#
# Required env:
#   T3_CKPT_PATH        Checkpoint dir (contains config.json, ckpt.pt).
#
# Optional env:
#   T3_BASELINE_REF     Default HEAD~2 (the commit just before the axis 3 pair).
#   T3_LIMIT            lm_eval --limit. Default 50.
#   T3_OUT_DIR          Where logs + parked output dirs land. Default ./validation_axis3.
#   T3_GSM8K_GEN_LENGTH Forwarded to eval_t3.sh as GSM8K_GEN_LENGTH. Default 512.
#   T3_BLOCK_SIZE       Forwarded as BLOCK_SIZE. Default 6.
#
# Usage from the Think-Then-Talk repo root:
#   T3_CKPT_PATH=/path/to/state_2 bash scripts/compare_gsm8k.sh
#
# Aborts on dirty tree (eval_t3.sh would clobber uncommitted launch tweaks).
#
set -euo pipefail

: "${T3_CKPT_PATH:?T3_CKPT_PATH must be set to a checkpoint directory}"
T3_BASELINE_REF=${T3_BASELINE_REF:-HEAD~2}
T3_LIMIT=${T3_LIMIT:-50}
T3_OUT_DIR=${T3_OUT_DIR:-./validation_axis3}
T3_GSM8K_GEN_LENGTH=${T3_GSM8K_GEN_LENGTH:-512}
T3_BLOCK_SIZE=${T3_BLOCK_SIZE:-6}

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ -n $(git status --porcelain) ]]; then
  echo "[compare_gsm8k] FATAL: working tree is dirty. Stash or commit first." >&2
  echo "  hint: git stash push -u -m \"pre-gsm8k-compare\"" >&2
  exit 2
fi

CURRENT_REF=$(git rev-parse --abbrev-ref HEAD)
[[ "${CURRENT_REF}" == "HEAD" ]] && CURRENT_REF=$(git rev-parse HEAD)

mkdir -p "${T3_OUT_DIR}"
echo "[compare_gsm8k] outputs -> ${T3_OUT_DIR}"
echo "[compare_gsm8k] baseline: ${T3_BASELINE_REF}"
echo "[compare_gsm8k] current : ${CURRENT_REF}"
echo "[compare_gsm8k] ckpt    : ${T3_CKPT_PATH}"
echo "[compare_gsm8k] limit   : ${T3_LIMIT}"

cleanup() {
  local now
  now=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
  if [[ "${now}" != "${CURRENT_REF}" ]]; then
    echo "[compare_gsm8k] restoring branch ${CURRENT_REF}"
    git checkout -q "${CURRENT_REF}" || true
  fi
}
trap cleanup EXIT

# eval_t3.sh writes to a fixed output_path that includes the checkpoint dir
# basename, but does NOT include the git ref. We park the dir after each run
# so the second run doesn't clobber the first.
EVAL_OUT_ROOT="evals_results/gsm8k-${T3_GSM8K_GEN_LENGTH}"

run_one() {
  local label=$1                      # "baseline" or "current"
  local ref=$2
  local log_path="${T3_OUT_DIR}/gsm8k_${label}.log"
  local park_dir="${T3_OUT_DIR}/gsm8k_${label}_evals_results"

  echo
  echo "=========================================================================="
  echo "[compare_gsm8k] running gsm8k --limit ${T3_LIMIT} on ${label} (${ref})"
  echo "=========================================================================="
  git checkout -q "${ref}"

  rm -rf "${EVAL_OUT_ROOT}"
  set +e
  CKPT_PATH="${T3_CKPT_PATH}" \
  GSM8K_GEN_LENGTH="${T3_GSM8K_GEN_LENGTH}" \
  BLOCK_SIZE="${T3_BLOCK_SIZE}" \
    bash eval_t3.sh gsm8k --limit "${T3_LIMIT}" 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[compare_gsm8k] ${label} run failed with exit ${rc}; see ${log_path}" >&2
    exit ${rc}
  fi

  rm -rf "${park_dir}"
  if [[ -d "${EVAL_OUT_ROOT}" ]]; then
    mv "${EVAL_OUT_ROOT}" "${park_dir}"
    echo "[compare_gsm8k] parked ${label} eval outputs at ${park_dir}"
  else
    echo "[compare_gsm8k] WARN: expected ${EVAL_OUT_ROOT} not found" >&2
  fi
}

run_one baseline "${T3_BASELINE_REF}"
run_one current  "${CURRENT_REF}"

echo
echo "=========================================================================="
echo "[compare_gsm8k] exact_match scores (parsed from the lm_eval summary table)"
echo "=========================================================================="

# lm_eval prints a markdown table near the end of its run with rows like:
#   |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5000|±  |0.0707|
# We grab every gsm8k row that mentions exact_match. Two rows usually appear
# (strict-match and flexible-extract); both matter.
extract_em() {
  local log=$1
  if [[ ! -f "${log}" ]]; then
    echo "  (no log at ${log})"
    return
  fi
  grep -E '\|gsm8k\|.*exact_match' "${log}" || echo "  (no exact_match line found in ${log})"
}

echo "-- baseline (${T3_BASELINE_REF}) --"
extract_em "${T3_OUT_DIR}/gsm8k_baseline.log"
echo
echo "-- current  (${CURRENT_REF}) --"
extract_em "${T3_OUT_DIR}/gsm8k_current.log"
echo

echo "[compare_gsm8k] full logs:"
echo "  ${T3_OUT_DIR}/gsm8k_baseline.log"
echo "  ${T3_OUT_DIR}/gsm8k_current.log"
echo "[compare_gsm8k] full eval output dirs (samples + results.json):"
echo "  ${T3_OUT_DIR}/gsm8k_baseline_evals_results/"
echo "  ${T3_OUT_DIR}/gsm8k_current_evals_results/"
echo
echo "[compare_gsm8k] gate: |Δ exact_match| ≤ 0.02 (2pp). If exceeded, stop and"
echo "                investigate before treating axis 3 as validated."

#!/usr/bin/env bash
#
# eval_t3.sh — lm_eval harness wrapper for Think-Then-Talk.
#
# Picks a task preset, builds the JSON `--model_args` payload, and runs
# `eval_t3.py` (which registers the `t3_model` and `t3_model_infer` models
# with lm_eval; both wrap `T3DecodeEngine` from `model/inference_engine.py`).
#
# Required environment variables:
#   CKPT_PATH                 Path to a saved T3 checkpoint dir (contains config.json).
#
# Optional knobs (all have sensible defaults below):
#   TASK                      First positional arg, or this var. One of:
#                             gsm8k | math500 | hendrycks_math500 | minerva_math500 | humaneval
#   CUDA_NO                   Index of the GPU to expose. Default: 0.
#   DEVICE                    Override of `cuda:${CUDA_NO}`. Pass "auto" to let
#                             T3InferenceModel resolve LOCAL_RANK/WORLD_SIZE.
#   MODEL_NAME                lm_eval registry name. Default: t3_model_infer
#                             (uses T3InferenceModel — disables activation
#                             checkpointing). Set to `t3_model` for the plain
#                             training-side runtime.
#   BLOCK_SIZE                Tokens decoded per think+talk iteration. Default 6.
#   GSM8K_GEN_LENGTH          GSM8K generation length. Default 512.
#   MATH500_GEN_LENGTH        MATH-500 generation length. Default 512.
#   HUMANEVAL_GEN_LENGTH      HumanEval generation length. Default 256.
#
# Reveal/decode policy overrides (folded into `model_config["denoise"]`).
# Behaviour gated by USE_SH_DENOISE_POLICY:
#   USE_SH_DENOISE_POLICY     "1" => every override below MUST be set; "0" =>
#                             ignore the overrides and use the checkpoint's
#                             saved policy. Default 1.
#   REVEAL_K                  Positions revealed per denoise step. int.
#   REVEAL_POLICY_MODE        random | greedy | ar_force (see eval/overrides.py).
#   REVEAL_RANDOM_WEIGHT      Mixture weight for random reveal. float in [0,1].
#   REVEAL_GREEDY_WEIGHT      Mixture weight for greedy reveal. float in [0,1].
#   REVEAL_AR_FORCE_WEIGHT    Mixture weight for ar_force reveal. float in [0,1].
#   DECODE_POLICY_MODE        fix | greedy. (Older "confidence_threshold" is
#                             treated as greedy by the harness.)
#   DECODE_FIX_K              For fix mode: tokens decoded per step.
#   DECODE_MAX_K              For greedy mode: max tokens per step.
#   DECODE_MIN_K              For greedy mode: min tokens per step.
#   DECODE_CONFIDENCE_THRESHOLD  Confidence floor used by greedy mode.
#   DECODE_FIX_WEIGHT         Mixture weight for fix decode.
#   DECODE_GREEDY_WEIGHT      Mixture weight for greedy decode.
#
# Pass-through to lm_eval CLI: any extra positional args after TASK are
# forwarded to `eval_t3.py` as-is (e.g. `--limit 50`).
#
set -euo pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ---- Task selection ----------------------------------------------------------
TASK=${1:-${TASK:-gsm8k}}
if [[ $# -gt 0 ]]; then
  shift
fi

# ---- Inference target --------------------------------------------------------
CKPT_PATH=${CKPT_PATH:?CKPT_PATH must be set to a checkpoint directory}
CUDA_NO=${CUDA_NO:-0}
DEVICE=${DEVICE:-cuda:${CUDA_NO}}
MODEL_NAME=${MODEL_NAME:-t3_model_infer}
BLOCK_SIZE=${BLOCK_SIZE:-6}
HUMANEVAL_GEN_LENGTH=${HUMANEVAL_GEN_LENGTH:-256}
GSM8K_GEN_LENGTH=${GSM8K_GEN_LENGTH:-512}
MATH500_GEN_LENGTH=${MATH500_GEN_LENGTH:-512}

EVAL_LAUNCHER=(python)

echo "[eval_t3.sh] launcher=${EVAL_LAUNCHER[*]} model=${MODEL_NAME} device=${DEVICE}"
echo "[eval_t3.sh] task=${TASK} ckpt=${CKPT_PATH}"

# ---- Denoise policy overrides ------------------------------------------------
USE_SH_DENOISE_POLICY=${USE_SH_DENOISE_POLICY:-1}

REVEAL_K=${REVEAL_K:-2}
REVEAL_POLICY_MODE=${REVEAL_POLICY_MODE:-greedy}
REVEAL_RANDOM_WEIGHT=${REVEAL_RANDOM_WEIGHT:-0}
REVEAL_GREEDY_WEIGHT=${REVEAL_GREEDY_WEIGHT:-1}
REVEAL_AR_FORCE_WEIGHT=${REVEAL_AR_FORCE_WEIGHT:-0}

DECODE_POLICY_MODE=${DECODE_POLICY_MODE:-greedy}
DECODE_FIX_K=${DECODE_FIX_K:-2}
DECODE_MAX_K=${DECODE_MAX_K:-2}
DECODE_MIN_K=${DECODE_MIN_K:-1}
DECODE_CONFIDENCE_THRESHOLD=${DECODE_CONFIDENCE_THRESHOLD:-0.9}
DECODE_FIX_WEIGHT=${DECODE_FIX_WEIGHT:-0}
DECODE_GREEDY_WEIGHT=${DECODE_GREEDY_WEIGHT:-1}

require_nonempty() {
  local var_name
  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      echo "[eval_t3.sh] Error: ${var_name} must be set when USE_SH_DENOISE_POLICY=1." >&2
      exit 1
    fi
  done
}

if [[ "${USE_SH_DENOISE_POLICY}" == "1" ]]; then
  require_nonempty \
    REVEAL_K REVEAL_POLICY_MODE REVEAL_RANDOM_WEIGHT REVEAL_GREEDY_WEIGHT REVEAL_AR_FORCE_WEIGHT \
    DECODE_POLICY_MODE DECODE_FIX_K DECODE_MAX_K DECODE_MIN_K DECODE_CONFIDENCE_THRESHOLD \
    DECODE_FIX_WEIGHT DECODE_GREEDY_WEIGHT
fi

# ---- JSON builder ------------------------------------------------------------
build_model_args_json() {
  local gen_length=$1
  local prompt_prefix=$2
  local prompt_suffix=$3

  python -c '
import json, sys
(
    ckpt_path, gen_length, block_size, device,
    prompt_prefix, prompt_suffix,
    use_sh_denoise_policy, reveal_k, reveal_policy_mode,
    reveal_random_weight, reveal_greedy_weight, reveal_ar_force_weight,
    decode_policy_mode, decode_fix_k, decode_max_k, decode_min_k,
    decode_confidence_threshold, decode_fix_weight, decode_greedy_weight,
) = sys.argv[1:]
print(json.dumps({
    "ckpt_path": ckpt_path,
    "gen_length": int(gen_length),
    "block_size": int(block_size),
    "device": device,
    "show_speed": True,
    "prompt_prefix": prompt_prefix,
    "prompt_suffix": prompt_suffix,
    "use_sh_denoise_policy": use_sh_denoise_policy == "1",
    "reveal_k": int(reveal_k),
    "reveal_policy_mode": reveal_policy_mode,
    "reveal_random_weight": float(reveal_random_weight),
    "reveal_greedy_weight": float(reveal_greedy_weight),
    "reveal_ar_force_weight": float(reveal_ar_force_weight),
    "decode_policy_mode": decode_policy_mode,
    "decode_fix_k": int(decode_fix_k),
    "decode_max_k": int(decode_max_k),
    "decode_min_k": int(decode_min_k),
    "decode_confidence_threshold": float(decode_confidence_threshold),
    "decode_fix_weight": float(decode_fix_weight),
    "decode_greedy_weight": float(decode_greedy_weight),
}))
' \
    "${CKPT_PATH}" "${gen_length}" "${BLOCK_SIZE}" "${DEVICE}" \
    "${prompt_prefix}" "${prompt_suffix}" \
    "${USE_SH_DENOISE_POLICY}" "${REVEAL_K}" "${REVEAL_POLICY_MODE}" \
    "${REVEAL_RANDOM_WEIGHT}" "${REVEAL_GREEDY_WEIGHT}" "${REVEAL_AR_FORCE_WEIGHT}" \
    "${DECODE_POLICY_MODE}" "${DECODE_FIX_K}" "${DECODE_MAX_K}" "${DECODE_MIN_K}" \
    "${DECODE_CONFIDENCE_THRESHOLD}" "${DECODE_FIX_WEIGHT}" "${DECODE_GREEDY_WEIGHT}"
}

# ---- Output-path helpers -----------------------------------------------------
sanitize_output_component() {
  local value=$1
  value=${value//\//__}
  value=${value// /_}
  value=${value//:/_}
  printf '%s' "${value}"
}

checkpoint_output_subdir() {
  local ckpt_dir
  local ckpt_state
  ckpt_dir=$(basename "$(dirname "${CKPT_PATH}")")
  ckpt_state=$(basename "${CKPT_PATH}")
  printf '%s__%s' \
    "$(sanitize_output_component "${ckpt_dir}")" \
    "$(sanitize_output_component "${ckpt_state}")"
}

# ---- Task launchers ----------------------------------------------------------
run_math_style_eval() {
  local task_name=$1
  local gen_length=$2
  local output_root=$3
  local output_dir="${output_root}/$(checkpoint_output_subdir)"
  local extra_args=("${@:4}")
  local prompt_prefix='Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.\n\n'
  local model_args_json
  model_args_json=$(build_model_args_json "${gen_length}" "${prompt_prefix}" "")

  "${EVAL_LAUNCHER[@]}" eval_t3.py \
    --model "${MODEL_NAME}" \
    --model_args "${model_args_json}" \
    --tasks "${task_name}" \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_path "${output_dir}" \
    --log_samples \
    "${extra_args[@]}"
}

case "${TASK}" in
  gsm8k)
    OUTPUT_ROOT="evals_results/gsm8k-${GSM8K_GEN_LENGTH}"
    run_math_style_eval gsm8k "${GSM8K_GEN_LENGTH}" "${OUTPUT_ROOT}" "$@"
    SAMPLE_FILE=$(find "${OUTPUT_ROOT}" -type f -name 'samples_gsm8k_*.jsonl' | sort | tail -n 1)
    python postprocess.py "${SAMPLE_FILE}"
    ;;
  math_500|math500|hendrycks_math500)
    run_math_style_eval hendrycks_math500 "${MATH500_GEN_LENGTH}" "evals_results/math500-${MATH500_GEN_LENGTH}" "$@"
    ;;
  minerva_math500)
    run_math_style_eval minerva_math500 "${MATH500_GEN_LENGTH}" "evals_results/minerva_math500-${MATH500_GEN_LENGTH}" "$@"
    ;;
  humaneval)
    MODEL_ARGS_JSON=$(build_model_args_json "${HUMANEVAL_GEN_LENGTH}" "" "")
    "${EVAL_LAUNCHER[@]}" eval_t3.py \
      --tasks humaneval \
      --confirm_run_unsafe_code \
      --model "${MODEL_NAME}" \
      --model_args "${MODEL_ARGS_JSON}" \
      --num_fewshot 0 \
      --batch_size 1 \
      --output_path "evals_results/humaneval-${HUMANEVAL_GEN_LENGTH}/$(checkpoint_output_subdir)" \
      --log_samples \
      "$@"
    ;;
  *)
    echo "[eval_t3.sh] Error: unsupported TASK=${TASK}. Supported: gsm8k, hendrycks_math500, minerva_math500, humaneval" >&2
    exit 1
    ;;
esac

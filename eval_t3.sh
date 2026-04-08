#!/usr/bin/env bash
set -euo pipefail

# Simple manual presets.
# Uncomment the block you want to run and keep the others commented.

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

TASK=${1:-${TASK:-gsm8k}}
if [[ $# -gt 0 ]]; then
  shift
fi

CKPT_PATH=${CKPT_PATH:-/share2/home/tliu0205/dllm/Think-Then-Talk/cfg6-p1-l30-r32-a64-lm-t6-mlp-7.5lr-2-denoise-mix-6-90-2/state_2}
CUDA_NO=${CUDA_NO:-3}
BLOCK_SIZE=${BLOCK_SIZE:-6}
HUMANEVAL_GEN_LENGTH=${HUMANEVAL_GEN_LENGTH:-256}
GSM8K_GEN_LENGTH=${GSM8K_GEN_LENGTH:-512}
MATH500_GEN_LENGTH=${MATH500_GEN_LENGTH:-512}
USE_NEW_INFERENCE_FRAMEWORK=${USE_NEW_INFERENCE_FRAMEWORK:-1}

if [[ "${USE_NEW_INFERENCE_FRAMEWORK}" == "1" ]]; then
  MODEL_NAME=${MODEL_NAME:-t3_model_infer}
  THINK_DEVICE1=${THINK_DEVICE1:-cuda:${CUDA_NO}}
  THINK_DEVICE2=${THINK_DEVICE2:-cuda:${CUDA_NO}}
  TALK_DEVICE=${TALK_DEVICE:-cuda:${CUDA_NO}}
else
  MODEL_NAME=${MODEL_NAME:-t3_model}
  THINK_DEVICE1=${THINK_DEVICE1:-cuda:${CUDA_NO}}
  THINK_DEVICE2=${THINK_DEVICE2:-cuda:${CUDA_NO}}
  TALK_DEVICE=${TALK_DEVICE:-cuda:${CUDA_NO}}
fi

EVAL_LAUNCHER=(python)

echo "[eval_t3.sh] launcher=${EVAL_LAUNCHER[*]} model=${MODEL_NAME}"
echo "[eval_t3.sh] task=${TASK}"
echo "[eval_t3.sh] devices: think_device1=${THINK_DEVICE1}, think_device2=${THINK_DEVICE2}, talk_device=${TALK_DEVICE}"
echo "[eval_t3.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Denoise policy source:
# - USE_SH_DENOISE_POLICY=0: use the original reveal/decode policy from ${CKPT_PATH}/config.json
# - USE_SH_DENOISE_POLICY=1: use only the shell values below
#   If this mode is on, any empty reveal/decode value is treated as an error.
USE_SH_DENOISE_POLICY=${USE_SH_DENOISE_POLICY:-1}

# Reveal policy modes (choose one with REVEAL_POLICY_MODE):
# - random: reveal candidate positions randomly
# - greedy: reveal the highest-confidence candidate positions
# - ar_force: reveal from left to right
# Reveal mixture weights are only for advanced runs. One reveal mode is sampled per step.
REVEAL_K=${REVEAL_K:-2}
REVEAL_POLICY_MODE=${REVEAL_POLICY_MODE:-greedy}
REVEAL_RANDOM_WEIGHT=${REVEAL_RANDOM_WEIGHT:-0}
REVEAL_GREEDY_WEIGHT=${REVEAL_GREEDY_WEIGHT:-1}
REVEAL_AR_FORCE_WEIGHT=${REVEAL_AR_FORCE_WEIGHT:-0}

# Decode policy modes (choose one with DECODE_POLICY_MODE):
# - fix: always decode exactly DECODE_FIX_K tokens
# - greedy: count confident positions, then clamp the result into [DECODE_MIN_K, DECODE_MAX_K]
# Older checkpoint modes like greedy_threshold / confidence_threshold are treated as greedy.
# Decode weights are mixture weights. They decide which one of the 2 decode modes is sampled.
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

POLICY_MODEL_ARGS=",use_sh_denoise_policy=${USE_SH_DENOISE_POLICY}"
if [[ "${USE_SH_DENOISE_POLICY}" == "1" ]]; then
  require_nonempty \
    REVEAL_K REVEAL_POLICY_MODE REVEAL_RANDOM_WEIGHT REVEAL_GREEDY_WEIGHT REVEAL_AR_FORCE_WEIGHT \
    DECODE_POLICY_MODE DECODE_FIX_K DECODE_MAX_K DECODE_MIN_K DECODE_CONFIDENCE_THRESHOLD \
    DECODE_FIX_WEIGHT DECODE_GREEDY_WEIGHT
  POLICY_MODEL_ARGS+=",reveal_k=${REVEAL_K},reveal_policy_mode=${REVEAL_POLICY_MODE},reveal_random_weight=${REVEAL_RANDOM_WEIGHT},reveal_greedy_weight=${REVEAL_GREEDY_WEIGHT},reveal_ar_force_weight=${REVEAL_AR_FORCE_WEIGHT},decode_policy_mode=${DECODE_POLICY_MODE},decode_fix_k=${DECODE_FIX_K},decode_max_k=${DECODE_MAX_K},decode_min_k=${DECODE_MIN_K},decode_confidence_threshold=${DECODE_CONFIDENCE_THRESHOLD},decode_fix_weight=${DECODE_FIX_WEIGHT},decode_greedy_weight=${DECODE_GREEDY_WEIGHT}"
fi

build_model_args_json() {
  local gen_length=$1
  local prompt_prefix=$2
  local prompt_suffix=$3

  /home/tliu0205/miniconda3/envs/fast-dllm/bin/python -c '
import json
import sys

(
    ckpt_path,
    gen_length,
    block_size,
    think_device1,
    think_device2,
    talk_device,
    prompt_prefix,
    prompt_suffix,
    use_sh_denoise_policy,
    reveal_k,
    reveal_policy_mode,
    reveal_random_weight,
    reveal_greedy_weight,
    reveal_ar_force_weight,
    decode_policy_mode,
    decode_fix_k,
    decode_max_k,
    decode_min_k,
    decode_confidence_threshold,
    decode_fix_weight,
    decode_greedy_weight,
) = sys.argv[1:]

print(json.dumps({
    "ckpt_path": ckpt_path,
    "gen_length": int(gen_length),
    "block_size": int(block_size),
    "think_device1": think_device1,
    "think_device2": think_device2,
    "talk_device": talk_device,
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
    "${CKPT_PATH}" "${gen_length}" "${BLOCK_SIZE}" "${THINK_DEVICE1}" "${THINK_DEVICE2}" "${TALK_DEVICE}" \
    "${prompt_prefix}" "${prompt_suffix}" "${USE_SH_DENOISE_POLICY}" "${REVEAL_K}" "${REVEAL_POLICY_MODE}" \
    "${REVEAL_RANDOM_WEIGHT}" "${REVEAL_GREEDY_WEIGHT}" "${REVEAL_AR_FORCE_WEIGHT}" "${DECODE_POLICY_MODE}" \
    "${DECODE_FIX_K}" "${DECODE_MAX_K}" "${DECODE_MIN_K}" "${DECODE_CONFIDENCE_THRESHOLD}" \
    "${DECODE_FIX_WEIGHT}" "${DECODE_GREEDY_WEIGHT}"
}

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
    run_math_style_eval gsm8k "${GSM8K_GEN_LENGTH}" "/share2/home/tliu0205/dllm/Think-Then-Talk/evals_results/gsm8k-${GSM8K_GEN_LENGTH}" "$@"
    SAMPLE_FILE=$(find "/share2/home/tliu0205/dllm/Think-Then-Talk/evals_results/gsm8k-${GSM8K_GEN_LENGTH}" -type f -name 'samples_gsm8k_*.jsonl' | sort | tail -n 1)
    python /share2/home/tliu0205/dllm/Think-Then-Talk/postprocess.py "${SAMPLE_FILE}"
    ;;
  math_500|math500|hendrycks_math500)
    run_math_style_eval hendrycks_math500 "${MATH500_GEN_LENGTH}" "/share2/home/tliu0205/dllm/Think-Then-Talk/evals_results/math500-${MATH500_GEN_LENGTH}" "$@"
    ;;
  minerva_math500)
    run_math_style_eval minerva_math500 "${MATH500_GEN_LENGTH}" "/share2/home/tliu0205/dllm/Think-Then-Talk/evals_results/minerva_math500-${MATH500_GEN_LENGTH}" "$@"
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
      --output_path "/share2/home/tliu0205/dllm/Think-Then-Talk/evals_results/humaneval-${HUMANEVAL_GEN_LENGTH}/$(checkpoint_output_subdir)" \
      --log_samples \
      "$@"
    ;;
  *)
    echo "[eval_t3.sh] Error: unsupported TASK=${TASK}. Supported: gsm8k, hendrycks_math500, minerva_math500, humaneval" >&2
    exit 1
    ;;
esac

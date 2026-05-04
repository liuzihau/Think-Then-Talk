#!/usr/bin/env bash
# 2x H200 via DDP (data parallel only — no pipeline, no ZeRO).
# Effective batch size doubles vs. single-GPU:
#   total = train_micro_batch_size_per_gpu x num_processes x gradient_accumulation_steps
# Adjust LR / accumulation steps in the JSON config if you want a fair comparison.
set -euo pipefail

CONFIG="${CONFIG:-train/config/r32/p1_t6_r32-Nemo-chat-math-code-gsm8k-more-math.json}"
SAVEDIR="${SAVEDIR:-0}"

accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 2 \
    --multi_gpu \
    train_h200.py \
    --training_config "$CONFIG" \
    --savedir "$SAVEDIR"

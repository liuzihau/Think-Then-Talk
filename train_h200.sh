#!/usr/bin/env bash
# Single H200 (default).
set -euo pipefail

CONFIG="${CONFIG:-train/config/r32/p1_t6_r32-Nemo-chat-math-code-gsm8k-more-math.json}"
SAVEDIR="${SAVEDIR:-0}"

accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 1 \
    train_h200.py \
    --training_config "$CONFIG" \
    --savedir "$SAVEDIR"

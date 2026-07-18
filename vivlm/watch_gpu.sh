#!/usr/bin/env bash
# vivlm/watch_gpu.sh [out_csv] [interval_s] — ghi util/VRAM song song với train.
# Chạy trong tmux pane riêng: bash vivlm/watch_gpu.sh vivlm/outputs/gpu_log.csv 30
OUT="${1:-vivlm/outputs/gpu_log.csv}"
INTERVAL="${2:-30}"
mkdir -p "$(dirname "$OUT")"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv,noheader,nounits -l "$INTERVAL" >> "$OUT"

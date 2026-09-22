#!/usr/bin/env bash
# Train only the proposed SWFMO + residual-flow method at five resolutions.
# Required: CACHE_ROOT and RUN_ROOT. Optional: EPOCHS, MANIFEST_NAME, PYTHON.
set -euo pipefail

: "${CACHE_ROOT:?Set CACHE_ROOT to the five resolution caches}"
: "${RUN_ROOT:?Set RUN_ROOT for checkpoints and logs}"
PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-40}
WORKERS=${WORKERS:-2}
MANIFEST_NAME=${MANIFEST_NAME:-strict_id_ood_splits_x120t81.json}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
GPU_INDEX=${CUDA_VISIBLE_DEVICES%%,*}
GPU_INDEX=${GPU_INDEX:-0}
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

for resolution in 120 240 480 960 1024; do
  cache_dir="$CACHE_ROOT/cache_x${resolution}_t81"
  manifest="$cache_dir/$MANIFEST_NAME"
  [[ -f "$cache_dir/metadata.json" ]] || { echo "Missing cache: $cache_dir" >&2; exit 1; }
  [[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 1; }
  case "$resolution" in
    120) batch_size=16 ;;
    240) batch_size=12 ;;
    480) batch_size=6 ;;
    960) batch_size=3 ;;
    1024) batch_size=2 ;;
  esac

  base_dir="$RUN_ROOT/train_x${resolution}/swfmo"
  if [[ ! -f "$base_dir/result.json" ]]; then
    "$PYTHON" -u -m pdebench_1d.train \
      --model swfmo --cache-dir "$cache_dir" \
      --split-manifest "$MANIFEST_NAME" --output-dir "$base_dir" \
      --epochs "$EPOCHS" --batch-size "$batch_size" --workers "$WORKERS" \
      --val-every 5 --val-cases 726 --flow-steps 4 \
      --learning-rate 5e-4 --weight-decay 1e-4 \
      --shock-weight 2.0 --selection-objective global_l2re --resume
  else
    echo "SKIP completed $resolution/SWFMO"
  fi

  residual_dir="$RUN_ROOT/ours_x${resolution}/residual"
  if [[ -f "$residual_dir/result.json" ]]; then
    echo "SKIP completed $resolution/residual"
    continue
  fi
  "$PYTHON" -u -m pdebench_1d.train_residual_flow \
    --cache-dir "$cache_dir" --split-manifest "$MANIFEST_NAME" \
    --base-checkpoint "$base_dir/best.pt" --base-model swfmo \
    --output-dir "$residual_dir" --epochs "$EPOCHS" \
    --batch-size "$batch_size" --workers "$WORKERS" --val-every 2 --val-cases 726 \
    --time-samples 20 --base-steps 2 --residual-steps 2 \
    --learning-rate 5e-4 --weight-decay 1e-4 \
    --shock-weight 2.0 --endpoint-weight 1.0 --gradient-weight 0.05 \
    --wavelet-weight 0.1 --overshoot-weight 0.05 \
    --selection-objective shock_visual --resume
done

echo "SWFMO + residual outputs are under $RUN_ROOT"

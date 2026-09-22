#!/usr/bin/env bash
# Evaluate the proposed method at one or more native resolutions.
# Required: CACHE_ROOT, RUN_ROOT, and RESULT_ROOT.
set -euo pipefail


: "${CACHE_ROOT:?Set CACHE_ROOT to the five resolution caches}"
: "${RUN_ROOT:?Set RUN_ROOT to the trained checkpoints}"
: "${RESULT_ROOT:?Set RESULT_ROOT for metrics and figures}"
PYTHON=${PYTHON:-python}
MANIFEST_NAME=${MANIFEST_NAME:-strict_id_ood_splits_x120t81.json}
SPLIT=${SPLIT:-test}
BATCH_SIZE=${BATCH_SIZE:-8}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
GPU_INDEX=${CUDA_VISIBLE_DEVICES%%,*}
GPU_INDEX=${GPU_INDEX:-0}
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

mkdir -p "$RESULT_ROOT"
for resolution in ${RESOLUTIONS:-120 240 480 960 1024}; do
  cache_dir="$CACHE_ROOT/cache_x${resolution}_t81"
  "$PYTHON" -u -m pdebench_1d.evaluate_ours \
    --cache-dir "$cache_dir" --split-manifest "$MANIFEST_NAME" --split "$SPLIT" \
    --base-checkpoint "$RUN_ROOT/train_x${resolution}/swfmo/best.pt" \
    --residual-checkpoint "$RUN_ROOT/ours_x${resolution}/residual/best.pt" \
    --output "$RESULT_ROOT/ours_x${resolution}_${SPLIT}.json" \
    --batch-size "$BATCH_SIZE"
done

echo "Ours metrics are under $RESULT_ROOT"

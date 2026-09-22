#!/usr/bin/env bash
# Render a four-time pressure/gradient panel for one proposed-model checkpoint.
set -euo pipefail

: "${CACHE_DIR:?Set CACHE_DIR to one prepared cache}"
: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to a SWFMO best.pt}"
: "${RESIDUAL_CHECKPOINT:?Set RESIDUAL_CHECKPOINT to a residual best.pt}"
: "${OUTPUT:?Set OUTPUT to the PNG destination}"
PYTHON=${PYTHON:-python}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"$PYTHON" -u -m pdebench_1d.visualize_ours \
  --cache-dir "$CACHE_DIR" --split-manifest "${MANIFEST_NAME:-splits.json}" \
  --split "${SPLIT:-test}" --base-checkpoint "$BASE_CHECKPOINT" \
  --residual-checkpoint "$RESIDUAL_CHECKPOINT" --case-index "${CASE_INDEX:-0}" \
  --output "$OUTPUT"

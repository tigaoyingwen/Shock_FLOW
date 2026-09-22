#!/usr/bin/env bash
# Build the five native-resolution caches and one shared strict split manifest.
# Usage: SOURCE_HDF5=/path/to/data.hdf5 CACHE_ROOT=/path/to/caches bash "$0"
set -euo pipefail

: "${SOURCE_HDF5:?Set SOURCE_HDF5 to the PDEBench HDF5 file}"
: "${CACHE_ROOT:?Set CACHE_ROOT to a writable cache directory}"
PYTHON=${PYTHON:-python}
BATCH_SIZE=${CACHE_BATCH_SIZE:-16}
MANIFEST_NAME=${MANIFEST_NAME:-strict_id_ood_splits_x120t81.json}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for resolution in 120 240 480 960 1024; do
  cache_dir="$CACHE_ROOT/cache_x${resolution}_t81"
  "$PYTHON" -u -m pdebench_1d.prepare_cache \
    --source "$SOURCE_HDF5" \
    --output-dir "$cache_dir" \
    --spatial-size "$resolution" \
    --time-stride 1 \
    --spatial-reduction linear \
    --batch-size "$BATCH_SIZE"
done

source_cache="$CACHE_ROOT/cache_x120_t81"
"$PYTHON" -u -m pdebench_1d.create_strict_splits \
  --cache-dir "$source_cache" \
  --output-name "$MANIFEST_NAME"

for resolution in 120 240 480 960 1024; do
  cache_dir="$CACHE_ROOT/cache_x${resolution}_t81"
  if [[ "$cache_dir" != "$source_cache" ]]; then
    cp "$source_cache/$MANIFEST_NAME" "$cache_dir/$MANIFEST_NAME"
  fi
done

echo "Prepared caches under $CACHE_ROOT"
echo "Shared manifest: $source_cache/$MANIFEST_NAME"

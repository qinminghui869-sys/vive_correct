#!/usr/bin/env bash
# Run from any directory; all relative paths are resolved from the repository root.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/run_cv.sh [DATASET=samples] [OUT=outputs/cv_full] [extra run options]"
  echo "Runs rectified-only CV tracking + features; writes OUT/run.log; no models or repair."
  exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cv_dataset="${1:-samples}"
cv_output="${2:-outputs/cv_full}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
mkdir -p "$cv_output"
# Append on resume; pipefail propagates Python failures through tee.
python3 -u -m vive_correct run \
  --dataset "$cv_dataset" --out "$cv_output" --jobs 4 --coarse-stride 15 \
  "$@" --profile cv --stage all --model-review off 2>&1 | tee -a "$cv_output/run.log"

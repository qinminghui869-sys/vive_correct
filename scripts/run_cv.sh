#!/usr/bin/env bash
# Run from any directory; all relative paths are resolved from the repository root.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/run_cv.sh [DATASET] [OUT] [extra run options]"
  echo "DATASET default: /mnt/nas/synnas/脑电/KS3"
  echo "OUT default: outputs/ks3"
  echo "Bounded fast feature selection; writes OUT/run.log; no models or repair."
  echo "Use --feature-search full for exhaustive recovery; add --stage all for identity tracking."
  exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cv_dataset="${1:-/mnt/nas/synnas/脑电/KS3}"
cv_output="${2:-outputs/ks3}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
mkdir -p "$cv_output"
# Append on resume; pipefail propagates Python failures through tee.
python3 -u -m vive_correct run \
  --dataset "$cv_dataset" --out "$cv_output" --jobs 4 --coarse-stride 15 \
  --stage features "$@" --profile cv --model-review off 2>&1 | tee -a "$cv_output/run.log"

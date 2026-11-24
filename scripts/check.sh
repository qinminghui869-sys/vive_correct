#!/usr/bin/env bash
# The same checks run locally and in CI. No dataset or model credentials required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/check_docs.py
python3 -m ruff check .
python3 -m ruff format --check .
node --check vive_correct/review/review_ui/review.js
node --check vive_correct/review/review_ui/axis_rotation.js
python3 -m unittest discover -s tests -v
PYTHONPATH="ego_data:.:${PYTHONPATH:-}" python3 -m unittest discover -s ego_data/tests -v

#!/usr/bin/env bash
# Fetch nanolm at the pinned commit and install the Python dependencies.
#   bash scripts/setup.sh            (from the repository root; uses the active Python)
set -euo pipefail
cd "$(dirname "$0")/.."
NANOLM_COMMIT=8418fc1bae85657190f11749483bf2e729b9702c
if [ ! -d external/DeepDataMiningLearning/.git ]; then
  git clone --filter=blob:none https://github.com/lkk688/DeepDataMiningLearning.git external/DeepDataMiningLearning
fi
git -C external/DeepDataMiningLearning checkout -q "$NANOLM_COMMIT"
python -m pip install -q -r requirements.txt

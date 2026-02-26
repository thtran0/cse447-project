#!/usr/bin/env bash
set -euo pipefail

TEST_DATA="$1"
PRED_OUT="$2"

# Ensure work dir exists (it will be mounted to /job/work by docker run)
mkdir -p /job/work

# Train if needed (Checkpoint 1 safe)
if [ ! -f /job/work/model.json ]; then
  echo "Model not found — training..."
  python3 /job/src/myprogram.py train --work_dir /job/work
fi

python3 /job/src/myprogram.py test \
  --work_dir /job/work \
  --test_data "$TEST_DATA" \
  --test_output "$PRED_OUT"

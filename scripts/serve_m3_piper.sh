#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export JAX_PLATFORMS=cpu
exec .venv/bin/python scripts/serve_subtask_policy.py \
  --checkpoint checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500 \
  --host 127.0.0.1 --port 8000 "$@"

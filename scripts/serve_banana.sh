#!/usr/bin/env bash
# Launch the pi05_aloha_banana policy server.
# Run from the repo root with: bash scripts/serve_banana.sh
set -e
cd "$(dirname "$0")/.."

CKPT_STEP=${CKPT_STEP:-5000}
PORT=${PORT:-8000}
PROMPT=${PROMPT:-"put banana in the green plate"}

JAX_PLATFORMS=cpu python scripts/serve_policy.py \
    --env ALOHA \
    --default-prompt "${PROMPT}" \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config pi05_aloha_banana \
    --policy.dir "checkpoints/pi05_aloha_banana/banana_baseline/${CKPT_STEP}"

#!/usr/bin/env bash
# Launch the ALOHA banana policy server.
# Run from the repo root with: bash scripts/serve_banana.sh
set -e
cd "$(dirname "$0")/.."

CKPT_STEP=${CKPT_STEP:-5000}
PORT=${PORT:-8000}
PROMPT=${PROMPT:-"put banana in the green plate"}
POLICY_CONFIG=${POLICY_CONFIG:-pi05_aloha_banana}
POLICY_DIR=${POLICY_DIR:-checkpoints/pi05_aloha_banana/banana_baseline/${CKPT_STEP}}
# POLICY_CONFIG=${POLICY_CONFIG:-pi05_aloha_banana_subtask_segment}
# POLICY_DIR=${POLICY_DIR:-checkpoints/pi05_aloha_banana_subtask_segment/banana_subtask_seg_lr5e5/${CKPT_STEP}}
# POLICY_CONFIG=${POLICY_CONFIG:-pi05_aloha_banana_subgoal_base}
# POLICY_DIR=${POLICY_DIR:-checkpoints/pi05_aloha_banana_subgoal_base/banana_subgoal_base_lr5e5/${CKPT_STEP}}
# POLICY_CONFIG=${POLICY_CONFIG:-pi05_aloha_banana_traj}
# POLICY_DIR=${POLICY_DIR:-checkpoints/pi05_aloha_banana_traj/banana_traj_lr5e5/${CKPT_STEP}}

JAX_PLATFORMS=cpu python scripts/serve_policy_pytorch.py \
    --env ALOHA \
    --default-prompt "${PROMPT}" \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${POLICY_DIR}"

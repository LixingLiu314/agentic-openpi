#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  DATASET_PATH="$1"
else
  DATASET_PATH="${DATASET_PATH:-}"
fi

if [[ -z "${DATASET_PATH}" ]]; then
  echo "Usage: DATASET_PATH=playground/Datasets/<dataset_name> $0" >&2
  echo "   or: $0 playground/Datasets/<dataset_name>" >&2
  exit 2
fi

TRAJ_JSON="${TRAJ_JSON:-${DATASET_PATH}/trajectory_data/fk_bimanual.json}"
COT_JSON="${COT_JSON:-${DATASET_PATH}/trajectory_data/cot_text_prompts.json}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
COT_WORKERS="${COT_WORKERS:-16}"

python tools/trajectory/generate_aloha_fk_trajectory.py \
  --dataset-path "${DATASET_PATH}" \
  --include-arms both \
  --output "${TRAJ_JSON}"

python tools/trajectory/generate_aloha_bimanual_cot_prompts.py \
  --dataset_path "${DATASET_PATH}" \
  --traj_json "${TRAJ_JSON}" \
  --output_path "${COT_JSON}" \
  --image_size 256 \
  --left_gripper_dim 6 \
  --right_gripper_dim 13 \
  --num_workers "${COT_WORKERS}" \
  --interval 30

python tools/trajectory/visualize_all_aloha_fk_trajectories.py \
  --dataset-path "${DATASET_PATH}" \
  --traj-json "${TRAJ_JSON}" \
  --arm both \
  --no-skip-existing \
  --num-workers "${VIDEO_WORKERS}"

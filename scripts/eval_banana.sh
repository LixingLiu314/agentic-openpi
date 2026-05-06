#!/usr/bin/env bash
# Launch the piper_main client for pi05_aloha_banana evaluation.
# Run on the robot machine with: bash scripts/eval_banana.sh
set -e
cd "$(dirname "$0")/.."

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
NUM_EPISODES=${NUM_EPISODES:-3}
MAX_STEPS=${MAX_STEPS:-1000}
ACTION_HORIZON=${ACTION_HORIZON:-25}

source /opt/ros/noetic/setup.bash
source "$HOME/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m examples.aloha_real.piper_main \
    --host "${HOST}" \
    --port "${PORT}" \
    --action_horizon "${ACTION_HORIZON}" \
    --num_episodes "${NUM_EPISODES}" \
    --max_episode_steps "${MAX_STEPS}" \
    "$@"

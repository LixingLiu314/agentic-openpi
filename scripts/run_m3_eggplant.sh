#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/ros/noetic/setup.bash
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate aloha
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
export PYTHONPATH="$PWD/packages/openpi-client/src:${PYTHONPATH:-}"
exec python scripts/run_subtask_piper.py \
  --host 127.0.0.1 --port 8000 \
  --prompt "Put the eggplant into the box" \
  --output "logs/inference/$(date +%Y%m%d_%H%M%S_%N)" \
  "$@"

#!/usr/bin/env bash
# Launch the Aloha eval GUI.
#
# Usage:
#   bash scripts/start_aloha_eval_gui.sh [--mode basic|traj|subtask|subgoal] [--host HOST] [--port PORT] [--task "TASK"]
#
# Pre-reqs:
#   1.  Robot stack (Piper ROS) — usually already running on this host.
#   2.  ForeAct server (subgoal mode only) :
#         on 10.1.119.68 ->  python server_foreact.py
#   3.  Policy server is started by the GUI when "start local server" is checked.
#
# Keyboard:  1/2/3/4 = subtask key (mode 3), Space = pause, H = 回零, Q = quit.

set -e
cd "$(dirname "$0")/.."

# ROS env (best-effort)
[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash || true
[ -f "$HOME/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash" ] \
    && source "$HOME/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash" || true

export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH"

exec python -m examples.aloha_real.eval_gui.eval_gui "$@"

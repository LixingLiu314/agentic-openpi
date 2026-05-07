#!/usr/bin/env bash
# Headless CLI fallback (no PyQt) — runs a single mode end-to-end.
#
# Usage:
#   bash scripts/eval_aloha_modes.sh --mode basic|traj|subtask|subgoal [--host HOST] [--port PORT]
#
# This launches the GUI in --headless mode is *not* supported; we just start
# the GUI and let the user drive. For a truly headless run use
# `python -m examples.aloha_real.eval_gui.eval_runner_cli` (see README).

set -e
cd "$(dirname "$0")/.."
exec bash scripts/start_aloha_eval_gui.sh "$@"

#!/bin/bash
# Sequential subgoal experiments, runs after any current train_pytorch.py finishes.
# Mirrors the structure of run_banana_sweep.sh.
#
# Usage:
#   bash scripts/run_subgoal_sweep.sh [--start N]
#
#   --start N  Skip to experiment N (1-indexed). Default: 1.

set -euo pipefail

START_FROM=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_FROM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TORCHRUN="torchrun --standalone --nnodes=1 --nproc_per_node=8"
TRAIN="scripts/train_pytorch.py"
LOG_DIR="./sweep_logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local idx="$1"
    local exp_name="$2"
    shift 2
    local args=("$@")

    if [[ "$idx" -lt "$START_FROM" ]]; then
        echo "[skip] Experiment $idx: $exp_name"
        return
    fi

    echo ""
    echo "============================================================"
    echo "Experiment $idx: $exp_name"
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    local log_file="$LOG_DIR/${idx}_${exp_name}.log"
    $TORCHRUN $TRAIN "${args[@]}" --exp_name "$exp_name" 2>&1 | tee "$log_file"

    echo "------------------------------------------------------------"
    echo "Done: $exp_name  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "------------------------------------------------------------"
}

# Wait for any currently running training to finish
if pgrep -f "train_pytorch.py" > /dev/null 2>&1; then
    echo "Waiting for current train_pytorch.py to finish..."
    while pgrep -f "train_pytorch.py" > /dev/null 2>&1; do
        sleep 30
    done
    echo "Current training finished. Starting subgoal sweep at experiment $START_FROM."
else
    echo "No training in progress. Starting subgoal sweep at experiment $START_FROM."
fi

# ── Experiment 1: trajectory COT ────────────────────────────────
run_exp 1 "obstacle_baseline" \
    pi05_aloha_obstacle_baseline \
    --debug_steps 5

# ── Experiment 2: subgoal base camera only ─────────────────────
run_exp 2 "obstacle_subtask" \
    pi05_aloha_obstacle_subtask \
    --debug_steps 5

run_exp 3 "obstacle_traj" \
    pi05_aloha_obstacle_traj \
    --debug_steps 5

run_exp 4 "obstacle_subgoal" \
    pi05_aloha_obstacle_subgoal \
    --debug_steps 5

run_exp 5 "obstacle_all_cot" \
    pi05_aloha_obstacle_all_cot \
    --debug_steps 5

echo ""
echo "All subgoal experiments finished: $(date '+%Y-%m-%d %H:%M:%S')"

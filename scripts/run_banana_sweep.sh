#!/bin/bash
# Sequential hyperparameter sweep for banana subtask experiments.
# Run this from the project root in a separate terminal/pane.
# It waits for any currently running train_pytorch.py to finish before starting.
#
# Usage:
#   bash scripts/run_banana_sweep.sh [--start N]
#
#   --start N  Skip to experiment N (1-indexed). Default: 1.
#              Useful to resume after a failure.

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
    echo "Current training finished. Starting sweep at experiment $START_FROM."
else
    echo "No training in progress. Starting sweep at experiment $START_FROM."
fi

# ── Experiment 1: segment baseline (full fine-tune) ───────────────────────────
run_exp 1 "banana_subtask_segment" \
    pi05_aloha_banana_subtask_segment \
    --debug_steps 5

# ── Experiment 2: freeze PaliGemma VLM, train action expert only ──────────────
run_exp 2 "banana_subtask_seg_freeze_vlm" \
    pi05_aloha_banana_subtask_segment_freeze_vlm \
    --debug_steps 5

# ── Experiment 3: LR = 1e-5 (lower than baseline 2.5e-5) ─────────────────────
run_exp 3 "banana_subtask_seg_lr1e5" \
    pi05_aloha_banana_subtask_segment \
    --lr_schedule.peak_lr 1e-5 \
    --lr_schedule.decay_lr 1e-6 \
    --debug_steps 5

# ── Experiment 4: LR = 5e-5 (higher than baseline 2.5e-5) ────────────────────
run_exp 4 "banana_subtask_seg_lr5e5" \
    pi05_aloha_banana_subtask_segment \
    --lr_schedule.peak_lr 5e-5 \
    --lr_schedule.decay_lr 5e-6 \
    --debug_steps 5

# ── Experiment 5: 10k steps (2x baseline) ────────────────────────────────────
run_exp 5 "banana_subtask_seg_10k" \
    pi05_aloha_banana_subtask_segment \
    --num_train_steps 10000 \
    --lr_schedule.decay_steps 10000 \
    --save_interval 1000 \
    --debug_steps 5

echo ""
echo "All experiments finished: $(date '+%Y-%m-%d %H:%M:%S')"

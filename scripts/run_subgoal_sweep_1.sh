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

# Step 1: sleep 3 hours before checking GPU availability
echo "Sleeping 3 hours before checking GPU availability... (until $(date -d '+3 hours' '+%Y-%m-%d %H:%M:%S'))"
sleep 10800

# Step 2: poll every 5 minutes until all 8 GPUs each have >= 60 GB free
FREE_THRESHOLD_MIB=61440   # 60 GiB in MiB

check_gpu_free() {
    # Returns 0 (success) if ALL 8 GPUs have >= FREE_THRESHOLD_MIB free
    local counts
    counts=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | wc -l)
    if [[ "$counts" -lt 8 ]]; then
        return 1  # fewer than 8 GPUs visible
    fi
    while IFS= read -r free_mib; do
        if [[ "$free_mib" -lt "$FREE_THRESHOLD_MIB" ]]; then
            return 1
        fi
    done < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    return 0
}

echo "Waiting for all 8 GPUs to have >= 60 GB free (checking every 5 minutes)..."
while ! check_gpu_free; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU memory not sufficient yet, waiting 5 min..."
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
    sleep 300
done
echo "GPU memory condition met. Starting subgoal sweep at experiment $START_FROM."

run_exp 1 "eai_subtask" \
    pi05_aloha_eai_subtask \
    --debug_steps 5

run_exp 2 "eai_subgoal" \
    pi05_aloha_eai_subgoal \
    --debug_steps 5



echo ""
echo "All subgoal experiments finished: $(date '+%Y-%m-%d %H:%M:%S')"

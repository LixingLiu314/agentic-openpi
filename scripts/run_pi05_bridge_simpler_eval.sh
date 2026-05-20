#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/media/raid/workspace/xiahongyu/agentic-openpi
SIMPLER_ENV_ROOT=/media/raid/workspace/xiahongyu/SimplerEnv
SIMPLER_PYTHON=/media/raid/workspace/xiahongyu/miniconda3/envs/simpler_env/bin/python

ENABLE_DOUBAO_SUBTASK=${ENABLE_DOUBAO_SUBTASK:-0}
if [[ "${ENABLE_DOUBAO_SUBTASK}" == "1" ]]; then
  CKPT_PATH=${CKPT_PATH:-"${REPO_ROOT}/checkpoints/pi05_bridge_subtask/bridge_subtask_aligned/30000"}
  POLICY_CONFIG=${POLICY_CONFIG:-pi05_bridge_subtask}
else
  CKPT_PATH=${CKPT_PATH:-"${REPO_ROOT}/checkpoints/pi05_bridge/bridge_reproduce/30000"}
  POLICY_CONFIG=${POLICY_CONFIG:-pi05_bridge}
fi
PORT=${PORT:-18000}
CUDA_DEVICE=${CUDA_DEVICE:-6}
TASK=${TASK:-all}
EPISODES=${EPISODES:-96}
IMAGE_PREPROCESS=${IMAGE_PREPROCESS:-resize_square_224}
VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}
SKIP_RENDER_PRECHECK=${SKIP_RENDER_PRECHECK:-0}
DOUBAO_API_KEY=${DOUBAO_API_KEY:-${ARK_API_KEY:-${VOLCENKEY:-}}}
DOUBAO_BASE_URL=${DOUBAO_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}
DOUBAO_MODEL=${DOUBAO_MODEL:-doubao-seed-2-0-pro-260215}
COT_REFRESH_INTERVAL=${COT_REFRESH_INTERVAL:-6}
if [[ "${ENABLE_DOUBAO_SUBTASK}" == "1" ]]; then
  DEFAULT_EVAL_NAME=pi05_bridge_subtask_doubao
else
  DEFAULT_EVAL_NAME=pi05_bridge_reproduce_30000
fi

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-"${REPO_ROOT}/logs/simpler_eval/${DEFAULT_EVAL_NAME}/${RUN_ID}"}
RESULT_ROOT=${RESULT_ROOT:-"${REPO_ROOT}/results/simpler_eval/${DEFAULT_EVAL_NAME}/${RUN_ID}"}
COT_LOG_DIR=${COT_LOG_DIR:-"${RESULT_ROOT}/cot_logs"}

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

SERVER_LOG="${LOG_ROOT}/policy_server.log"
EVAL_LOG="${LOG_ROOT}/simpler_eval.log"
PID_FILE="${LOG_ROOT}/policy_server.pid"

cleanup() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

cd "${REPO_ROOT}"

echo "[run] repo=${REPO_ROOT}"
echo "[run] simpler_env_python=${SIMPLER_PYTHON}"
echo "[run] checkpoint=${CKPT_PATH}"
echo "[run] policy_config=${POLICY_CONFIG}"
echo "[run] port=${PORT}"
echo "[run] cuda_device=${CUDA_DEVICE}"
echo "[run] task=${TASK}"
echo "[run] episodes_per_task=${EPISODES}"
echo "[run] image_preprocess=${IMAGE_PREPROCESS}"
echo "[run] enable_doubao_subtask=${ENABLE_DOUBAO_SUBTASK}"
echo "[run] doubao_model=${DOUBAO_MODEL}"
echo "[run] cot_refresh_interval=${COT_REFRESH_INTERVAL}"
echo "[run] cot_log_dir=${COT_LOG_DIR}"
echo "[run] vk_icd_filenames=${VK_ICD_FILENAMES}"
echo "[run] log_root=${LOG_ROOT}"
echo "[run] result_root=${RESULT_ROOT}"

if [[ "${ENABLE_DOUBAO_SUBTASK}" == "1" && -z "${DOUBAO_API_KEY}" ]]; then
  echo "[run] ENABLE_DOUBAO_SUBTASK=1 requires DOUBAO_API_KEY, ARK_API_KEY, or VOLCENKEY" >&2
  exit 1
fi

if [[ "${SKIP_RENDER_PRECHECK}" != "1" ]]; then
  echo "[run] checking SAPIEN offscreen renderer..."
  PRECHECK_LOG="${LOG_ROOT}/sapien_offscreen_precheck.log"
  if ! CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    DISPLAY="" \
    VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
    "${SIMPLER_PYTHON}" -m sapien.example.offscreen \
    > "${PRECHECK_LOG}" 2>&1; then
    echo "[run] SAPIEN offscreen precheck failed; see ${PRECHECK_LOG}" >&2
    echo "[run] SimplerEnv RGB/video evaluation requires a working Vulkan renderer." >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
DISPLAY="" \
VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy_pytorch.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${POLICY_CONFIG}" \
  --policy.dir "${CKPT_PATH}" \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "[run] started policy server pid=${SERVER_PID}; log=${SERVER_LOG}"

echo "[run] waiting for policy server websocket..."
"${SIMPLER_PYTHON}" - <<PY
import os
import sys
import time

sys.path.insert(0, "${REPO_ROOT}/packages/openpi-client/src")
import websockets.sync.client
from openpi_client import msgpack_numpy

port = int("${PORT}")
pid = int(open("${PID_FILE}", "r", encoding="utf-8").read().strip())
deadline = time.time() + 900
while True:
    try:
        conn = websockets.sync.client.connect(
            f"ws://127.0.0.1:{port}",
            compression=None,
            max_size=None,
            open_timeout=5,
            ping_interval=None,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        conn.close()
        print("[run] policy server ready; metadata=", metadata, flush=True)
        break
    except Exception as exc:
        if time.time() > deadline:
            raise SystemExit(f"Timed out waiting for policy server: {exc}") from exc
        try:
            os.kill(pid, 0)
        except OSError as err:
            raise SystemExit(f"Policy server exited early; see ${SERVER_LOG}: {err}") from err
        time.sleep(5)
PY

echo "[run] starting SimplerEnv evaluation; log=${EVAL_LOG}"
EVAL_ARGS=(
  --simpler-env-root "${SIMPLER_ENV_ROOT}"
  --ckpt-path "${CKPT_PATH}"
  --logging-dir "${RESULT_ROOT}"
  --host 127.0.0.1
  --port "${PORT}"
  --task "${TASK}"
  --episodes "${EPISODES}"
  --image-preprocess "${IMAGE_PREPROCESS}"
  --renderer-device ""
  --additional-env-save-tags pi05_bridge_openpi
)

if [[ "${ENABLE_DOUBAO_SUBTASK}" == "1" ]]; then
  if [[ -z "${DOUBAO_API_KEY}" ]]; then
    echo "[run] ENABLE_DOUBAO_SUBTASK=1 requires DOUBAO_API_KEY, ARK_API_KEY, or VOLCENKEY" >&2
    exit 1
  fi
  EVAL_ARGS+=(
    --enable-doubao-subtask
    --doubao-api-key "${DOUBAO_API_KEY}"
    --doubao-base-url "${DOUBAO_BASE_URL}"
    --doubao-model "${DOUBAO_MODEL}"
    --cot-refresh-interval "${COT_REFRESH_INTERVAL}"
    --cot-log-dir "${COT_LOG_DIR}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
DISPLAY="" \
VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"${SIMPLER_PYTHON}" "${REPO_ROOT}/scripts/eval_pi05_bridge_simpler.py" \
  "${EVAL_ARGS[@]}" \
  > "${EVAL_LOG}" 2>&1

echo "[run] evaluation finished"
echo "[run] logs: ${LOG_ROOT}"
echo "[run] results: ${RESULT_ROOT}"

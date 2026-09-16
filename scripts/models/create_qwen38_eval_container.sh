#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.26.0rc1-openeuler}"
CONTAINER_NAME="${CONTAINER_NAME:-Qwen3.8-vllm_zt}"

BF16_DIR="${BF16_DIR:-/mnt/data1/models/Qwen3.8-27B}"
W4A8_DIR="${W4A8_DIR:-/mnt/data1/zt/qwen38-w4a8-test/output-layers28-35}"
W4A8_PATCH="${W4A8_PATCH:-/mnt/data1/zt/qwen38-w4a8-test/config/w4a8.py}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/data1/zt/benchmark/inference_eval_suite}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_DIR}/scripts/models}"
LOG_DIR="${LOG_DIR:-/mnt/data1/zt/qwen38-w4a8-test/logs-runtime}"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: required directory does not exist: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file does not exist: $path" >&2
    exit 1
  fi
}

require_dir "$BF16_DIR"
require_dir "$W4A8_DIR"
require_dir "$SCRIPT_DIR"
require_file "$W4A8_PATCH"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: Docker image is not available locally: $IMAGE" >&2
  exit 1
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "ERROR: container already exists: $CONTAINER_NAME" >&2
  echo "Inspect it with: docker inspect $CONTAINER_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

docker run -dit \
  --name "$CONTAINER_NAME" \
  --shm-size=16g \
  --network host \
  --ipc host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e TASK_QUEUE_ENABLE=1 \
  -e OMP_NUM_THREADS=1 \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$BF16_DIR":/models/Qwen3.8-27B:ro \
  -v "$W4A8_DIR":/models/Qwen3.8-27B-W4A8-L28-35:ro \
  -v "$W4A8_PATCH":/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py:ro \
  -v "$SCRIPT_DIR":/workspace/scripts:ro \
  -v "$LOG_DIR":/workspace/logs \
  "$IMAGE" bash

echo
echo "Container created: $CONTAINER_NAME"
echo "BF16 model:        /models/Qwen3.8-27B"
echo "W4A8 model:        /models/Qwen3.8-27B-W4A8-L28-35"
echo "Scripts:           /workspace/scripts"
echo "Logs:              /workspace/logs"
echo
echo "Start BF16:"
echo "  docker exec -d $CONTAINER_NAME bash /workspace/scripts/start_bf16.sh"
echo "Start W4A8:"
echo "  docker exec -d $CONTAINER_NAME bash /workspace/scripts/start_w4a8.sh"


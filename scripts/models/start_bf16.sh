#!/usr/bin/env bash
set -euo pipefail

if pgrep -af "vllm serve" >/dev/null 2>&1; then
  echo "ERROR: a vLLM service is already running in this container." >&2
  pgrep -af "vllm serve" >&2 || true
  exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"

exec vllm serve /models/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B-BF16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  > /workspace/logs/bf16-vllm.log 2>&1


#!/usr/bin/env bash
set -euo pipefail

mapfile -t pids < <(pgrep -f "vllm serve" || true)
if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No vLLM service is running in this container."
  exit 0
fi

echo "Sending SIGTERM to vLLM process(es): ${pids[*]}"
kill -TERM "${pids[@]}"

for _ in $(seq 1 60); do
  if ! pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "vLLM stopped cleanly."
    exit 0
  fi
  sleep 1
done

echo "ERROR: vLLM did not stop within 60 seconds; no SIGKILL was sent." >&2
exit 1

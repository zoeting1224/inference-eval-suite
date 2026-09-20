#!/usr/bin/env bash
set -e

export IMAGE=quay.io/ascend/vllm-ascend:v0.26.0rc1-openeuler

docker run -dit \
  --name Qwen3.8-vllm_zt \
  --shm-size=16g \
  --net=host \
  --ipc=host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /mnt/data1/models:/models \
  -v /mnt/data1/zt/benchmark/inference_eval_suite:/workspace/inference_eval_suite \
  -v /mnt/data1/zt/qwen38-w4a8-test:/workspace/qwen38-w4a8-test \
  -v /mnt/data1/zt/qwen38-w4a8-test/config/w4a8.py:/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py:ro \
  "$IMAGE" bash

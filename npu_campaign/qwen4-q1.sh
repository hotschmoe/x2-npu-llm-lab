#!/bin/bash
# q1: matmul bandwidth anatomy on Qwen decode (profile mode): real vs NODOT (DMA only) vs NODMA (compute only) vs both
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s1'
for m in 0 14 15 16; do
  env_x=""; [ $m != 0 ] && env_x="GGML_HEXAGON_OPS650_SPOOF=$m"
  powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-anat-m$m -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "$env_x" -TimeoutSec 400 2>&1 | tail -3
done

#!/bin/bash
# q8: DMA shape experiments (spoofed: NODOT, outputs garbage): 2D tile rows (14) vs 1D contiguous tile rows (17); xp kernels on for the expert ops
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s6'
for m in 14 17; do
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-dma-m$m -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "GGML_HEXAGON_OPS650_SPOOF=$m;GGML_HEXAGON_MV_XPIPE=1" -TimeoutSec 300 2>&1 | tail -2
done

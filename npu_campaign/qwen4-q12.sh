#!/bin/bash
# q12: s8 = Q4C (contiguous weight DMA, q4_0c dots) + XPIPE: gate first, then profile.
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s8'
K="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=1"
bash npu_campaign/qwen4-gate.sh s8 "$P" "$K" plain hello
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-q4c-prof -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "$K" -TimeoutSec 400 2>&1 | tail -2

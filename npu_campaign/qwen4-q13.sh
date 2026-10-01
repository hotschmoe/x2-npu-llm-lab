#!/bin/bash
# q13: s9 = s8 + Q8_0 contiguous (GGML_HEXAGON_Q4C=3): gate then profile
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s9'
K="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3"
bash npu_campaign/qwen4-gate.sh s9 "$P" "$K" plain hello
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-q4c3-prof -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "$K" -TimeoutSec 400 2>&1 | tail -2

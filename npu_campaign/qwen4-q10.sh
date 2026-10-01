#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s7'
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-gdn-prof -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_GDN_RB=1" -TimeoutSec 400 2>&1 | tail -2
bash npu_campaign/qwen4-gate.sh s7 "$P" "GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_GDN_RB=1" plain hello

#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s1'
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-anat-m9 -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "GGML_HEXAGON_OPS650_SPOOF=9" -TimeoutSec 300 2>&1 | tail -2

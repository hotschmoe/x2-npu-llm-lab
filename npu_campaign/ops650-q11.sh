#!/bin/bash
# ops650 queue 11: profile of s10 (flush WQ threshold 32 KB) and of s9 (4 KB) for a direct comparison of the src/dst flush segments; then gate of s10.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
for v in s9 s10; do
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-seg-$v -Spec none -Tokens 24 -DeviceProfile 1 -Pkg "npu_campaign\pkg-ops650-$v" -ExtraEnv "GGML_HEXAGON_OPS650_SPOOF_CYCLE=0,0,0" -Prompt Hello > /dev/null 2>&1
done

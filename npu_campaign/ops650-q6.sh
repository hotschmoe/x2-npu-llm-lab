#!/bin/bash
# ops650 queue 6: spoof modes cycled inside ONE run (3 tokens per mode), plain decode, profile=1 (pkg s5 has per-op segment timing). Risky modes last.
cd /c/Users/hotschmoe/documents/llm_inference
for r in 1 2; do
python npu_campaign/lockrun650.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-cycB-r$r -Spec none -Tokens 60 -DeviceProfile 1 -Pkg 'npu_campaign\pkg-ops650-s5' -ExtraEnv "GGML_HEXAGON_OPS650_SPOOF_CYCLE=0,1,11,3,2,12,13,0,4,6,8,9,10,0" -Prompt Hello
done

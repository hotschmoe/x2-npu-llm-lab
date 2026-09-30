#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
python npu_campaign/lockrun650.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-seg-r1 -Spec none -Tokens 30 -DeviceProfile 1 -Pkg 'npu_campaign\pkg-ops650-s6' -ExtraEnv "GGML_HEXAGON_OPS650_SPOOF_CYCLE=0,1,0,11,0" -Prompt Hello

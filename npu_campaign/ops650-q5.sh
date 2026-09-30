#!/bin/bash
# ops650 queue 5: trace-mode (PROFILE=3) plain decode, per-stage events inside ops.
cd /c/Users/hotschmoe/documents/llm_inference
python npu_campaign/lockrun650.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-trace-plain -Spec none -Tokens 8 -DeviceProfile 3 -Pkg 'npu_campaign\pkg-ops650-s3' -ExtraEnv "GGML_HEXAGON_OPBATCH=1400;GGML_HEXAGON_OPQUEUE=2;GGML_HEXAGON_OPTRACE=32768" -Prompt Hello

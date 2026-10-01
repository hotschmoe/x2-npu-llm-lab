#!/bin/bash
# q14: s10 = s9 + GDN snapshot DMA. All opt-ins on: MV_XPIPE, Q4C=3, GDN_SNAPDMA. MTP gate (verify M=2 target + draft rows) then MTP profile.
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s10'
K="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3;GGML_HEXAGON_GDN_SNAPDMA=1"
bash npu_campaign/qwen4-gate.sh s10 "$P" "$K" mtp hello
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-s10-mtpprof -Spec mtp -DraftMax 1 -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "$K;LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536" -TimeoutSec 400 2>&1 | tail -2

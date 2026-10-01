#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
LR="python npu_campaign/lockrun.py --"; PS="powershell -NoProfile -File"; FCDV='LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536'
$LR $PS npu_campaign/run-mtp-speed.ps1 -Label m3-prof-plain -Spec none -Tokens 24 -DeviceProfile 1
$LR $PS npu_campaign/run-mtp-speed.ps1 -Label m3-prof-mtp -Spec mtp -DraftMax 1 -Tokens 24 -DeviceProfile 1 -ExtraEnv "$FCDV" -MtpProfile

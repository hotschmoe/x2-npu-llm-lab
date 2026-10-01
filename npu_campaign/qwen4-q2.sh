#!/bin/bash
# q2: MV_XPIPE (cross-item weight DMA pipeline for M=1 expert matmuls): profile + bit-exact plain-logit gate vs installed pkg-final2-f1
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s3'
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-xp1-prof -Spec none -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "GGML_HEXAGON_MV_XPIPE=1" -TimeoutSec 400 2>&1 | tail -2
powershell -NoProfile -File npu_campaign/mtp-gate.ps1 -Prefix q4-g1 -RefPkg 'npu_campaign\pkg-final2-f1' -CandPkg "$P" -Arms 'ref-plain,cand-plain' -CandExtraEnv "GGML_HEXAGON_MV_XPIPE=1" 2>&1 | tail -5 | tee -a npu_campaign/qwen4-gates.out

#!/bin/bash
# q7: s5 (xp incl. M=2..4 nx) MTP verify profiles xpipe 0/1 (Hello 32 tokens, fc+dv65536 d1)
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s5'
for x in 0 1; do
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4-mtpprof-xp$x -Spec mtp -DraftMax 1 -Tokens 40 -DeviceProfile 1 -Pkg "$P" -ExtraEnv "GGML_HEXAGON_MV_XPIPE=$x;LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536" -TimeoutSec 400 2>&1 | tail -2
done
powershell -NoProfile -File npu_campaign/mtp-gate.ps1 -Prefix q4-g3 -RefPkg 'npu_campaign\pkg-final2-f1' -CandPkg "$P" -Arms 'ref-plain,cand-plain,ref-mtp,cand-mtp' -CandExtraEnv "GGML_HEXAGON_MV_XPIPE=1;LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536" 2>&1 | tail -8 | tee -a npu_campaign/qwen4-gates.out

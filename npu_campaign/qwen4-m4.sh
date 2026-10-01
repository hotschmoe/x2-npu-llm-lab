#!/bin/bash
# master queue 4 (item 5 leftovers): prefill A/B of the tracker fix (pkg-merged2-m1 vs pkg-final2-f1, 3 rounds x pp1024/ub1024) and OPBATCH 1280 vs 3072 decode A/B (3 rounds).
# Each A/B runs inside ONE lockrun hold (NPU_LOCK_HELD=1 for the child scripts) behind the qwen4-guard pre-flight.
cd /c/Users/hotschmoe/documents/llm_inference
source npu_campaign/qwen4-lib.sh
wait_ac
python npu_campaign/lockrun.py -- python npu_campaign/qwen4-guard.py -- powershell -NoProfile -File npu_campaign/ab-multi.ps1 -Specs 'merged2|npu_campaign\pkg-merged2-m1\lib|npu_campaign\pkg-merged2-m1in|','final2|npu_campaign\pkg-final2-f1\lib|npu_campaign\pkg-final2-f1in|' -Rounds 2 -PP 1024 -UB 1024 -Reps 3 -Tag q4pfa 2>&1 | tail -8 | tee -a npu_campaign/qwen4-item5.out
python npu_campaign/lockrun.py -- python npu_campaign/qwen4-guard.py -- powershell -NoProfile -File npu_campaign/ab-multi.ps1 -Specs 'final2|npu_campaign\pkg-final2-f1\lib|npu_campaign\pkg-final2-f1in|','merged2|npu_campaign\pkg-merged2-m1\lib|npu_campaign\pkg-merged2-m1in|' -Rounds 1 -PP 1024 -UB 1024 -Reps 3 -Tag q4pfb 2>&1 | tail -8 | tee -a npu_campaign/qwen4-item5.out
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then ord="A B"; else ord="B A"; fi
  bash npu_campaign/qwen4-hold.sh "for a in $ord; do if [ \$a = A ]; then q4run q4-ob-3072-r$r 'npu_campaign\pkg-final2-f1' none 256 Hello 'X=1'; else q4run q4-ob-1280-r$r 'npu_campaign\pkg-final2-f1' none 256 Hello 'GGML_HEXAGON_OPBATCH=1280'; fi; done" | tee -a npu_campaign/qwen4-item5.out
done

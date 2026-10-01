#!/bin/bash
# master queue 3 (serial, lockrun-protected): s12 (pair-unrolled contiguous Q4_0/Q8_0 kernels). Profiles, gates, plain ABBA.
source npu_campaign/qwen4-lib.sh
OUT=npu_campaign/qwen4-m3.out
P='npu_campaign\pkg-qwen4-s12'
BASE='npu_campaign\pkg-final2-f1'
ALL="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3;GGML_HEXAGON_GDN_SNAPDMA=1"
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
# 1. plain gate (first, correctness before speed)
bash npu_campaign/qwen4-gate.sh s12 "$P" "$ALL" plain hello | tee -a $OUT
# 2. profiles
q4run q4-m3-prof-all  "$P" none 40 Hello "$ALL" -DeviceProfile 1 | tee -a $OUT
q4run q4-m3-prof-xp   "$P" none 40 Hello "GGML_HEXAGON_MV_XPIPE=1" -DeviceProfile 1 | tee -a $OUT
q4run q4-m3-mprof-all "$P" mtp 40 Hello "$ALL;$MT" -DraftMax 1 -DeviceProfile 1 | tee -a $OUT
# 3. MTP gate
bash npu_campaign/qwen4-gate.sh s12 "$P" "$ALL" mtp hello | tee -a $OUT
# 4. plain decode ABBA (3 rounds x 256 tokens): installed pkg-final2-f1 vs s12 all opt-ins
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then ord="A B"; else ord="B A"; fi
  bash npu_campaign/qwen4-hold.sh "for a in $ord; do if [ \$a = A ]; then q4run q4-m3-abba-base-r$r '$BASE' none 256 Hello 'X=1'; else q4run q4-m3-abba-cand-r$r '$P' none 256 Hello '$ALL'; fi; done" | tee -a $OUT
done

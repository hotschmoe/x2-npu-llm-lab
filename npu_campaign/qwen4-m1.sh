#!/bin/bash
# master queue 1 (serial, lockrun-protected): profiles + MTP gate + plain ABBA for the s10 candidate (all opt-ins).
source npu_campaign/qwen4-lib.sh
OUT=npu_campaign/qwen4-m1.out
P10='npu_campaign\pkg-qwen4-s10'
BASE='npu_campaign\pkg-final2-f1'
ALL="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3;GGML_HEXAGON_GDN_SNAPDMA=1"
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
# A. plain profile with all opt-ins (compare to q4-anat-m0 = 26.4 ms batch)
# (A done: q4-m1-prof-all)
# B. MTP verify profile: all opt-ins vs none
q4run q4-m1-mprof-all "$P10" mtp 40 Hello "$ALL;$MT" -DraftMax 1 -DeviceProfile 1 | tee -a $OUT
q4run q4-m1-mprof-off "$P10" mtp 40 Hello "X=1;$MT" -DraftMax 1 -DeviceProfile 1 | tee -a $OUT
# C. MTP gate (cand all opt-ins vs installed reference traces)
bash npu_campaign/qwen4-gate.sh s10all "$P10" "$ALL" mtp hello | tee -a $OUT
bash npu_campaign/qwen4-gate.sh s10all "$P10" "$ALL" plain hello | tee -a $OUT
# D. plain decode ABBA, 3 rounds x 256 tokens (base = installed pkg-final2-f1, cand = s10 all opt-ins); one hold per round (2 runs)
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then ord="A B"; else ord="B A"; fi
  bash npu_campaign/qwen4-hold.sh "for a in $ord; do if [ \$a = A ]; then q4run q4-m1-abba-base-r$r '$BASE' none 256 Hello 'X=1'; else q4run q4-m1-abba-cand-r$r '$P10' none 256 Hello '$ALL'; fi; done" | tee -a $OUT
done

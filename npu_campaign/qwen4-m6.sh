#!/bin/bash
# master queue 6: s13 = Q4_0/IQ4_NL tiles padded to 640 B in DDR (original dot kernels + one contiguous DMA per tile row) + xp + Q8c + SNAPDMA.
source npu_campaign/qwen4-lib.sh
OUT=npu_campaign/qwen4-m6.out
P='npu_campaign\pkg-qwen4-s13'
BASE='npu_campaign\pkg-final2-f1'
ALL="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3;GGML_HEXAGON_GDN_SNAPDMA=1"
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
bash npu_campaign/qwen4-gate.sh s13 "$P" "$ALL" plain hello | tee -a $OUT
q4run q4-m6-prof-all  "$P" none 40 Hello "$ALL" -DeviceProfile 1 | tee -a $OUT
q4run q4-m6-mprof-all "$P" mtp 40 Hello "$ALL;$MT" -DraftMax 1 -DeviceProfile 1 | tee -a $OUT
bash npu_campaign/qwen4-gate.sh s13 "$P" "$ALL" mtp hello | tee -a $OUT
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then ord="A B"; else ord="B A"; fi
  bash npu_campaign/qwen4-hold.sh "for a in $ord; do if [ \$a = A ]; then q4run q4-m6-abba-base-r$r '$BASE' none 256 Hello 'X=1'; else q4run q4-m6-abba-cand-r$r '$P' none 256 Hello '$ALL'; fi; done" | tee -a $OUT
done

#!/bin/bash
source npu_campaign/qwen4-lib.sh
OUT=npu_campaign/qwen4-m7.out
P12='npu_campaign\pkg-qwen4-s12'; BASE='npu_campaign\pkg-final2-f1'
ALL="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3;GGML_HEXAGON_GDN_SNAPDMA=1"
bash npu_campaign/qwen4-gate.sh s12 "$P12" "$ALL" mtp hello | tee -a $OUT
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then ord="A B"; else ord="B A"; fi
  bash npu_campaign/qwen4-hold.sh "for a in $ord; do if [ \$a = A ]; then q4run q4-m7-abba-base-r$r '$BASE' none 256 Hello 'X=1'; else q4run q4-m7-abba-s12-r$r '$P12' none 256 Hello '$ALL'; fi; done" | tee -a $OUT
done

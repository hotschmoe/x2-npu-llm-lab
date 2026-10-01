#!/bin/bash
# FINAL battery on pkg-qwen4-s14 (all qwen4 opts default ON, no env). One lockrun hold per step, guard inside.
source npu_campaign/qwen4-lib.sh
OUT=npu_campaign/qwen4-m8.out
P='npu_campaign\pkg-qwen4-s14'; BASE='npu_campaign\pkg-final2-f1'
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
G="python npu_campaign/lockrun.py -- python npu_campaign/qwen4-guard.py -- powershell -NoProfile -File"
# 1 PPL
$G npu_campaign/run-prefill-ppl.ps1 -Label q4-s14-ppl4 -LibDir $P/lib -BinDir $P/bin 2>&1 | tail -3 | tee -a $OUT
$G npu_campaign/run-prefill-ppl.ps1 -Label q4-s14-ppl16 -LibDir $P/lib -BinDir $P/bin -Chunks 16 2>&1 | tail -3 | tee -a $OUT
# 2 logits gates
bash npu_campaign/qwen4-gate.sh s14 "$P" "X=1" plain,mtp hello | tee -a $OUT
bash npu_campaign/qwen4-gate.sh s14 "$P" "X=1" plain,mtp code | tee -a $OUT
# 3 edge check
$G npu_campaign/run-fusion-edge-check.ps1 -LibDir $P/lib -BinDir $P/bin -WordsCsv 6,40,130,260,520 -Tag q4edge 2>&1 | tail -8 | tee -a $OUT
# 4 ABBA: configs x prompts x 3 rounds; one hold per (base,cand) pair, order alternates per round
for pr in Hello CODE; do
  for cfg in plain mtp pm; do
    for r in 1 2 3; do
      if [ $((r % 2)) = 1 ]; then ord="A B"; else ord="B A"; fi
      case $cfg in
        plain) sp=none; ex="X=1"; extra="";;
        mtp) sp=mtp; ex="$MT"; extra="-DraftMax 1";;
        pm) sp=mtp; ex="$MT"; extra="-DraftMax 1 -PMin 0.5";;
      esac
      if [ $pr = CODE ]; then PRT="$CODE"; else PRT="Hello"; fi
      bash npu_campaign/qwen4-hold.sh "for a in $ord; do if [ \$a = A ]; then q4run q4-m8-$pr-$cfg-base-r$r '$BASE' $sp 256 \"$PRT\" '$ex' $extra; else q4run q4-m8-$pr-$cfg-s14-r$r '$P' $sp 256 \"$PRT\" '$ex' $extra; fi; done" | tee -a $OUT
    done
  done
done

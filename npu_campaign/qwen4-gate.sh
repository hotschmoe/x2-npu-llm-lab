#!/bin/bash
# qwen4-gate.sh TAG PKG "ENV" [arms: plain,mtp] [prompt-tag: hello|code|sky]
# Bit-exact logits gate of candidate PKG (+ENV) vs the installed pkg-final2-f1 reference traces (kept in mtp-speed/q4ref-<prompt>-<arm>.logits.bin, created on first use).
# Appends results to qwen4-gates.out. Takes the NPU lock per run (run-mtp-speed.ps1).
cd /c/Users/hotschmoe/documents/llm_inference
TAG=$1; PKG=$2; ENVV=$3; ARMS=${4:-plain}; PT=${5:-hello}
source npu_campaign/m3_lib.sh >/dev/null 2>&1
LK="python npu_campaign/lockrun.py --"; [ "$NPU_LOCK_HELD" = 1 ] && LK=""
case $PT in hello) PR=Hello;; code) PR=$CODE;; sky) PR=$SKY;; esac
REF='npu_campaign\pkg-final2-f1'
D=npu_campaign/mtp-speed
for arm in ${ARMS//,/ }; do
  spec=none; [ $arm = mtp ] && spec=mtp
  refbin=$D/q4ref-$PT-$arm.logits.bin
  refenv="X=1"; [ $arm = mtp ] && refenv="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
  if [ ! -f $refbin ]; then
    $LK python npu_campaign/qwen4-guard.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4ref-$PT-$arm -Spec $spec -DraftMax 1 -Tokens 128 -Prompt "$PR" -LogitsTrace -Pkg "$REF" -ExtraEnv "$refenv" > /dev/null 2>&1
  fi
  cenv="$ENVV"; [ $arm = mtp ] && cenv="$ENVV;LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
  $LK python npu_campaign/qwen4-guard.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label q4g-$TAG-$PT-$arm -Spec $spec -DraftMax 1 -Tokens 128 -Prompt "$PR" -LogitsTrace -Pkg "$PKG" -ExtraEnv "$cenv" > /dev/null 2>&1
  out=$(python npu_campaign/compare_logits_traces.py $refbin $D/q4g-$TAG-$PT-$arm.logits.bin --require-identical 2>&1 | tail -2 | tr '\n' ' ')
  echo "GATE $TAG $PT $arm (target rows) exit=$? :: $out" | tee -a npu_campaign/qwen4-gates.out
  if [ $arm = mtp ]; then
    out=$(python npu_campaign/compare_logits_traces.py $refbin $D/q4g-$TAG-$PT-$arm.logits.bin --require-identical --context-index 1 --draft-greedy 2>&1 | tail -2 | tr '\n' ' ')
    echo "GATE $TAG $PT $arm (draft rows) exit=$? :: $out" | tee -a npu_campaign/qwen4-gates.out
  fi
  rm -f $D/q4g-$TAG-$PT-$arm.logits.bin
done

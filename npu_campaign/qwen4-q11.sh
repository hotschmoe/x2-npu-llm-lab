#!/bin/bash
# q11: OPPOLL A/B (plain decode, Hello 256, interleaved, 3 rounds, rotated order): off / spin(1) / predicted-sleep-then-spin(2). pkg s11 with xp + Q4C in every arm. One lockrun hold per round.
cd /c/Users/hotschmoe/documents/llm_inference
source npu_campaign/qwen4-lib.sh
wait_ac
P='npu_campaign\pkg-qwen4-s14'
K="X=1"
OUT=npu_campaign/qwen4-poll-abba.out
for r in 1 2 3; do
  case $r in 1) ord="off spin sleep";; 2) ord="sleep off spin";; 3) ord="spin sleep off";; esac
  bash npu_campaign/qwen4-hold.sh "for a in $ord; do case \$a in off) e='$K';; spin) e='$K;GGML_HEXAGON_OPPOLL=1';; sleep) e='$K;GGML_HEXAGON_OPPOLL=2';; esac; q4run q4-poll-\$a-r$r '$P' none 256 Hello \"\$e\"; done" | tee -a $OUT
done

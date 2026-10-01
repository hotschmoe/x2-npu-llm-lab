#!/bin/bash
# mtp3 queue 2: synthetic-acceptance ceiling (SPOOF, not a result), temp>0 acceptance, agentic prompts
cd /c/Users/hotschmoe/documents/llm_inference
LR="python npu_campaign/lockrun.py --"; PS="powershell -NoProfile -File"
P='llama.cpp-hexagon\pkg-wos'
FCDV='LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536'
tps() { grep -o '[0-9.]* tokens per second' npu_campaign/mtp-speed/$1.detail.log | tail -1; }
acc() { grep -o 'draft acceptance[^)]*)' npu_campaign/mtp-speed/$1.detail.log | tail -1; }
echo "== A. synthetic acceptance ceiling (SPOOF: output invalid), dmax 1, block poll"
for r in 1.0 0.85 0.5 0.0; do
  $LR $PS npu_campaign/run-mtp-speed.ps1 -Label m3q2-synth$r -Pkg $P -Spec mtp -DraftMax 1 -Tokens 256 -Prompt Hello -ExtraEnv "$FCDV" -ExtraArgs "--spec-synth-rates $r" > /dev/null 2>&1
  echo "synth $r $(tps m3q2-synth$r) $(acc m3q2-synth$r)"
done
CODE='Write a Python function that merges overlapping intervals, with a docstring and three unit tests.'
JSON='Give a JSON object for a tool call named get_weather with arguments city Paris and unit celsius, then explain each field.'
SKY='Explain why the sky appears blue in three clear sentences.'
echo "== B. agentic prompts, greedy: plain vs MTP (2 rounds)"
n=0
for pr in code json sky; do
  case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; esac
  for order in "none mtp" "mtp none"; do for s in $order; do n=$((n+1))
    if [ $s = mtp ]; then E="$FCDV"; else E=""; fi
    $LR $PS npu_campaign/run-mtp-speed.ps1 -Label m3q2-g-$pr-$s-$n -Pkg $P -Spec $s -DraftMax 1 -Tokens 256 -Prompt "$T" -ExtraEnv "$E" > /dev/null 2>&1
    echo "greedy $pr $s $(tps m3q2-g-$pr-$s-$n) $(acc m3q2-g-$pr-$s-$n)"
  done; done
done
echo "== C. temperature 0.6 top-p 0.95 top-k 20: acceptance and speed"
n=0
for pr in code json sky; do
  case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; esac
  for order in "none mtp" "mtp none"; do for s in $order; do n=$((n+1))
    if [ $s = mtp ]; then E="$FCDV"; else E=""; fi
    $LR $PS npu_campaign/run-mtp-speed.ps1 -Label m3q2-t-$pr-$s-$n -Pkg $P -Spec $s -DraftMax 1 -Tokens 256 -Prompt "$T" -ExtraEnv "$E" -ExtraArgs "--temp 0.6 --top-p 0.95 --top-k 20" > /dev/null 2>&1
    echo "temp0.6 $pr $s $(tps m3q2-t-$pr-$s-$n) $(acc m3q2-t-$pr-$s-$n)"
  done; done
done
echo "== done"

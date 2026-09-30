#!/bin/bash
# ops650 queue 13 (trimmed): interleaved untraced speed ABBA, plain decode Hello 128 tokens, 3 rounds with rotating order.
# variants: base = installed merged2-m1; s14 = tracker + vgather conv + WQ 32 KB (fusions OFF); s14F = s14 + every opt-in fusion (env mask 988160 + TOPK alias).
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
ALL="GGML_HEXAGON_OPS650_FUSE=988160;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
declare -A PK=( [base]='npu_campaign\pkg-merged2-m1' [s14]='npu_campaign\pkg-ops650-s14' [s14F]='npu_campaign\pkg-ops650-s14' )
declare -A EV=( [base]='X=1' [s14]='X=1' [s14F]="$ALL" )
ORDERS=( "base s14 s14F" "s14F s14 base" "s14 base s14F" )
r=0
for O in "${ORDERS[@]}"; do
  r=$((r+1))
  for k in $O; do
    $L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-ab2-$k-r$r -Spec none -Tokens 128 -Pkg "${PK[$k]}" -ExtraEnv "${EV[$k]}" -Prompt Hello > /dev/null 2>&1
  done
done

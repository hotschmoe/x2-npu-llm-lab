#!/bin/bash
# ops650 queue 17: interleaved untraced MTP speed ABBA (draft-max 1, fused catch-up, dv 65536, Hello 128 tokens), 3 rounds, rotating order.
# variants: base = installed merged2-m1; s14F = s14 + every opt-in fusion (mask 988160 incl. bit 19 small-T router top-k, MOE_TOPK alias).
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
ALL="GGML_HEXAGON_OPS650_FUSE=988160;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
declare -A PK=( [base]='npu_campaign\pkg-merged2-m1' [s14F]='npu_campaign\pkg-ops650-s14' )
declare -A EV=( [base]="$MT" [s14F]="$MT;$ALL" )
ORDERS=( "base s14F" "s14F base" "base s14F" )
r=0
for O in "${ORDERS[@]}"; do
  r=$((r+1))
  for k in $O; do
    $L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-abm-$k-r$r -Spec mtp -DraftMax 1 -Tokens 128 -Pkg "${PK[$k]}" -ExtraEnv "${EV[$k]}" -Prompt Hello > /dev/null 2>&1
  done
done

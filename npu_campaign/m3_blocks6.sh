#!/bin/bash
# mtp3 phase 6: find the first divergent draft row between fused catch-up (forced) and unfused at draft-max 2 (timing independent)
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = all ]; then shift; for b in "$@"; do echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks6.sh block $b; done; echo "== all done"; exit 0; fi
b=$2
case $b in
 fctrace) # traced logits (full vocab rows for target and draft), LLAMA_TRACE=1 prints per-step acceptance
   for v in nofc fc; do
     if [ $v = fc ]; then E="LLAMA_TRACE=1;LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_FC_FORCE=1"; else E="LLAMA_TRACE=1"; fi
     $PSF npu_campaign/run-mtp-speed.ps1 -Label m3o-$v -Pkg "$PE" -Spec mtp -DraftMax 2 -Tokens 96 -Prompt "$SKY" -LogitsTrace -ExtraEnv "$E" > /dev/null 2>&1
     echo "m3o-$v tps=$(tps m3o-$v) acc=$(acc m3o-$v)"
   done ;;
 *) echo "unknown block $b" ;;
esac

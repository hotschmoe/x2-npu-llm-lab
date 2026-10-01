#!/bin/bash
# mtp3 phase 2 (after merging ops650): new package pkg-mtp3-e. usage like m3_blocks.sh. NEVER edit this file while a driver runs; add m3_blocks3.sh instead.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
REF='npu_campaign\pkg-merged2-m1'
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = all ]; then shift; for b in "$@"; do echo "## block $b"; $LR "$BASH" npu_campaign/m3_blocks2.sh block $b; done; echo "== all done"; exit 0; fi
b=$2
case $b in
 gateH|gateS|gateC) # bit-exact gate: ref = merged2-m1, cand = mtp3-e (+fc), 128 traced tokens
   case $b in gateH) T=Hello;; gateS) T="$SKY";; gateC) T="$CODE";; esac
   $PSF npu_campaign/mtp-gate.ps1 -Prefix m3x-$b -RefPkg "$REF" -CandPkg "$PE" -Prompt "$T" -Tokens 128 -CandExtraEnv "LLAMA_MTP_FUSED_CATCHUP=1" 2>&1 | tail -12 ;;
 pplA) $PSF npu_campaign/run-prefill-ppl.ps1 -Label m3x-ppl4 -LibDir npu_campaign/pkg-mtp3-e/lib -BinDir npu_campaign/pkg-mtp3-e/bin 2>&1 | tail -2 ;;
 pplB) $PSF npu_campaign/run-prefill-ppl.ps1 -Label m3x-ppl16 -Chunks 16 -LibDir npu_campaign/pkg-mtp3-e/lib -BinDir npu_campaign/pkg-mtp3-e/bin 2>&1 | tail -2 ;;
 ceil1|ceil2) # SPOOF ceilings on the new base (block poll and spin), Hello256
   if [ $b = ceil1 ]; then ord="1.0 0.85 0.71"; else ord="0.71 0.85 1.0"; fi
   run m3y-$b-plain "$PE" none 1 256 Hello ""
   for r in $ord; do run m3y-$b-synth$r "$PE" mtp 1 256 Hello "$FCDV" "--spec-synth-rates $r"; done
   run m3y-$b-plainspin "$PE" none 1 256 Hello "GGML_HEXAGON_OPPOLL=1"
   run m3y-$b-synth1.0spin "$PE" mtp 1 256 Hello "$FCDV;GGML_HEXAGON_OPPOLL=1" "--spec-synth-rates 1.0" ;;
 fcd1|fcd2) # forced fused catch-up with draft-max 2 vs unfused, gated 0.8, on the new base
   if [ $b = fcd1 ]; then ord="nofc fc"; else ord="fc nofc"; fi
   for pr in rep json hello; do case $pr in rep) T="$REP";; hello) T=Hello;; json) T="$JSON";; esac
     for v in $ord; do case $v in nofc) run m3m-$b-$pr-$v "$PE" mtp 2 256 "$T" "LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" ;;
       fc) run m3m-$b-$pr-$v "$PE" mtp 2 256 "$T" "LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_FC_FORCE=1;LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" ;; esac; done; done ;;
 *) echo "unknown block $b" ;;
esac

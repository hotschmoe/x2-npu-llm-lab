#!/bin/bash
# temperature > 0 acceptance (PI-like sampler: temp 0.6, top-p 0.95, top-k 20), new base. Timing indifferent; speed recorded only if AC.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
b=$2
case $b in
 tp) for pr in code json sky rep; do case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; rep) T="$REP";; esac
       run m3t-$pr-plain "$PE" none 1 256 "$T" "" "$T06"
       run m3t-$pr-mtp1 "$PE" mtp 1 256 "$T" "$FCDV" "$T06"
       run m3t-$pr-mtp2 "$PE" mtp 2 256 "$T" "LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" "$T06"
     done ;;
esac

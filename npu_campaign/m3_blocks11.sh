#!/bin/bash
# mtp3 final: p-min sweep and the final interleaved ABBA blocks (AC only). NEVER edit while a driver runs.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
PW="$PKG_WOS"
b=$2
case $b in
 pm1) # first-token confidence gate on the new base (Hello, JSON): d1 fc+dv with p-min 0 / .5 / .7 / .85, plain for reference
   for pr in hello json; do case $pr in hello) T=Hello;; json) T="$JSON";; esac
     run m3p-$pr-plain "$PE" none 1 256 "$T" ""
     for pm in 0 0.5 0.7 0.85; do run m3p-$pr-pm$pm "$PE" mtp 1 256 "$T" "$FCDV" "" $pm; done
   done ;;
 fin-*) # fin-<prompt>-<round>
   pr=$(echo $b | cut -d- -f2); r=$(echo $b | cut -d- -f3)
   case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; rep) T="$REP";; hello) T=Hello;; esac
   case $r in 1) ord="A B C D E F";; 2) ord="F E D C B A";; 3) ord="D E F A B C";; esac
   G2="LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8"
   for v in $ord; do l=m3w-$pr-r$r-$v
     case $v in
      A) run $l "$PW" none 1 256 "$T" "" ;;                          # plain, installed
      B) run $l "$PE" none 1 256 "$T" "" ;;                          # plain, mtp3-e (ops650 merged)
      C) run $l "$PW" mtp 1 256 "$T" "$FCDV" ;;                      # MTP d1, installed
      D) run $l "$PE" mtp 1 256 "$T" "$FCDV" ;;                      # MTP d1, mtp3-e
      E) run $l "$PE" mtp 1 256 "$T" "$FCDV" "" $(cat npu_campaign/m3_pminx.txt) ;;     # MTP d1 + first-token p-min
      F) run $l "$PE" mtp 2 256 "$T" "$G2" ;;                        # MTP d2, 2nd token gated (p-min-next 0.8), fc auto-off
      G) run $l "$PE" mtp 2 256 "$T" "$G2" "" $(cat npu_campaign/m3_pminx.txt) ;;       # MTP d2 gated both
     esac; done ;;
esac

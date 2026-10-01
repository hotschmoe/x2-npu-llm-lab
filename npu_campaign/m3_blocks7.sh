#!/bin/bash
# mtp3 final interleaved ABBA blocks (AC only). NEVER edit while a driver runs.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
PW="$PKG_WOS"
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = all ] || [ "$1" = allac ]; then mode=$1; shift; for b in "$@"; do
  if [ $mode = allac ]; then wait_ac; fi
  echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks7.sh block $b; done; echo "== all done"; exit 0; fi
b=$2     # fin-<prompt>-<round>
pr=$(echo $b | cut -d- -f2); r=$(echo $b | cut -d- -f3)
case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; rep) T="$REP";; hello) T=Hello;; esac
arms="A B C D E F G H"
case $r in 1) ord="A B C D E F G H";; 2) ord="H G F E D C B A";; 3) ord="D E F G H A B C";; esac
G2="LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8"
SP="GGML_HEXAGON_OPPOLL=1"
for v in $ord; do
  l=m3z-$pr-r$r-$v
  case $v in
   A) run $l "$PW" none 1 256 "$T" "" ;;                                  # plain, installed
   B) run $l "$PE" none 1 256 "$T" "" ;;                                  # plain, mtp3-e (ops650 merged)
   C) run $l "$PW" mtp 1 256 "$T" "$FCDV" ;;                              # MTP d1, installed
   D) run $l "$PE" mtp 1 256 "$T" "$FCDV" ;;                              # MTP d1, mtp3-e
   E) run $l "$PE" mtp 2 256 "$T" "$G2" ;;                                # MTP d2 gated (p-min-next 0.8), fc auto-off
   F) run $l "$PE" mtp 1 256 "$T" "$FCDV;$SP" ;;                          # MTP d1 + spin
   G) run $l "$PE" mtp 2 256 "$T" "$G2;$SP" ;;                            # MTP d2 gated + spin
   H) run $l "$PE" none 1 256 "$T" "$SP" ;;                               # plain + spin
  esac
done

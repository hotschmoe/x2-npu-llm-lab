#!/bin/bash
# mtp3 phase 4: garbage hunt (timing independent) and final speed blocks. NEVER edit while a driver runs.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
PW="$PKG_WOS"
PBASE='npu_campaign\pkg-decode2-base'
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = all ] || [ "$1" = allac ]; then mode=$1; shift; for b in "$@"; do
  if [ $mode = allac ]; then wait_ac; fi
  echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks4.sh block $b; done; echo "== all done"; exit 0; fi
b=$2
case $b in
 gh-e-d1a|gh-e-d1b|gh-base-d1a|gh-base-d1b|gh-e-d2a|gh-e-d2b|gh-base-d2a|gh-base-d2b) # garbage hunt: multi-slot replay (4 slots) + p-min 0.7, 3 runs per block
   case $b in gh-e-*) P="$PE";; gh-base-*) P="$PBASE";; esac
   for i in 1 2 3; do
     if [[ $b == *-d1* ]]; then run m3i-$b-$i "$P" mtp 1 256 Hello "$FCDV;GGML_HEXAGON_REPLAY_SLOTS=4" "" 0.7
     else run m3i-$b-$i "$P" mtp 2 256 "$SKY" "LLAMA_MTP_FUSED_CATCHUP=1;GGML_HEXAGON_REPLAY_SLOTS=4" "" 0.7; fi
     echo "   slashes=$(grep -c '////' $D/m3i-$b-$i.log) words=$(wc -w < $D/m3i-$b-$i.log)"; done ;;
 *) echo "unknown block $b" ;;
esac

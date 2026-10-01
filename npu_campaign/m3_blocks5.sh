#!/bin/bash
# mtp3 phase 5: original failing mechanism: fused catch-up FORCED with draft-max 2 + multi-slot replay + p-min (timing independent). NEVER edit while a driver runs.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = all ]; then shift; for b in "$@"; do echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks5.sh block $b; done; echo "== all done"; exit 0; fi
b=$2
case $b in
 fc1|fc2|fc3|fc4) # 3 runs each: d2 p0.7 sky fc-forced slots4 ; two d1 p0.7 Hello slots4 runs interleaved
   for i in 1 2 3; do
     run m3n-$b-d2fc-$i "$PE" mtp 2 256 "$SKY" "LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_FC_FORCE=1;GGML_HEXAGON_REPLAY_SLOTS=4" "" 0.7
     echo "   slashes=$(grep -c '////' $D/m3n-$b-d2fc-$i.log) words=$(wc -w < $D/m3n-$b-d2fc-$i.log)"
   done ;;
 *) echo "unknown block $b" ;;
esac

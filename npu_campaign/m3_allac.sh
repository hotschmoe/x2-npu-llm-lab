#!/bin/bash
# AC-gated driver for m3_blocks2.sh blocks: waits for AC power before every block (polls every 60 s).
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
BASH="C:/Program Files/Git/usr/bin/bash.exe"
S=$1; shift
for b in "$@"; do wait_ac; echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/$S block $b; done; echo "== all done"

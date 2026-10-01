#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
BASH="C:/Program Files/Git/usr/bin/bash.exe"
while ! grep -q "== master done" npu_campaign/m3_p11.out 2>/dev/null; do sleep 60; done
wait_ac; echo "## block tp $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks10.sh block tp
echo "== done"

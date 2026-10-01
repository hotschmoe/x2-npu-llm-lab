#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
BASH="C:/Program Files/Git/usr/bin/bash.exe"
while ! grep -q "== master done" npu_campaign/m3_p11.out 2>/dev/null; do sleep 60; done
for b in fin-sky-1; do wait_ac; echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks11.sh block $b; sleep 45; done
echo "== done"

#!/bin/bash
# ONE sequential driver (fair to other agents): release the lock and sleep 45 s between blocks; speed blocks wait for AC.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
BASH="C:/Program Files/Git/usr/bin/bash.exe"
blk() { if [ "$3" = 1 ]; then wait_ac; fi; echo "## block $2 $(date +%H:%M)"; $LR "$BASH" npu_campaign/$1 block $2; sleep 45; }
blk m3_blocks4.sh gh-e-d2b 0
blk m3_blocks4.sh gh-e-d1b 0
blk m3_blocks9.sh pm1 1
for r in 1 2 3; do for pr in code json sky rep; do blk m3_blocks9.sh fin-$pr-$r 1; done; done
echo "== master done"

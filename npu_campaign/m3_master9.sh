#!/bin/bash
# ONE sequential driver (fair to other agents): release the lock and sleep 45 s between blocks; speed blocks wait for AC.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
BASH="C:/Program Files/Git/usr/bin/bash.exe"
blk() { # blk SCRIPT NAME SPEED(0/1)
  if [ "$3" = 1 ]; then wait_ac; fi
  echo "## block $2 $(date +%H:%M)"; $LR "$BASH" npu_campaign/$1 block $2; sleep 45; }
blk m3_blocks6.sh fctrace 0
blk m3_blocks4.sh gh-e-d2b 0
blk m3_blocks4.sh gh-e-d1b 0
blk m3_blocks5.sh fc2 0
for r in 1 2 3; do for pr in code json sky rep; do blk m3_blocks7.sh fin-$pr-$r 1; done; done
echo "== master done"

#!/bin/bash
# mtp3 phase 3: blocks3.sh all|allac NAME... ; allac waits for AC power before every block (speed blocks). NEVER edit while a driver runs.
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = all ] || [ "$1" = allac ]; then mode=$1; shift; for b in "$@"; do
  if [ $mode = allac ]; then wait_ac; fi
  echo "## block $b $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks3.sh block $b; done; echo "== all done"; exit 0; fi
b=$2
case $b in
 edge) $PSF npu_campaign/run-fusion-edge-check.ps1 -BinDir npu_campaign/pkg-mtp3-e/bin -LibDir npu_campaign/pkg-mtp3-e/lib -Tag m3x-edge 2>&1 | tail -10 ;;
 *) echo "unknown block $b" ;;
esac

#!/bin/bash
# ops650 queue 20: bisect the 16-chunk PPL deviation (13.3337 vs 13.3331 expected) on pkg s14.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
S14=npu_campaign/pkg-ops650-s14
run() { # label fuse alias
  GGML_HEXAGON_OPS650_FUSE=$2 GGML_HEXAGON_MOE_TOPK_ALIAS=$3 $L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label $1 -LibDir $S14/lib -BinDir $S14/bin -Chunks 16 > npu_campaign/$1.out 2>&1
}
run ops650-ppl16-off 0 0
run ops650-ppl16-fuse 988160 0
run ops650-ppl16-alias 0 1

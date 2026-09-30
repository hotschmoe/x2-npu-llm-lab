#!/bin/bash
# ops650 queue 21: pkg s16 (all gated fusions on by default, router top-k alias only for <=4 rows) 16-chunk PPL with NO env.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
S16=npu_campaign/pkg-ops650-s16
env -u GGML_HEXAGON_OPS650_FUSE -u GGML_HEXAGON_MOE_TOPK_ALIAS $L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-ppl16-s16def -LibDir $S16/lib -BinDir $S16/bin -Chunks 16 > npu_campaign/ops650-ppl16-s16def.out 2>&1

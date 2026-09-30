#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
S16=npu_campaign/pkg-ops650-s16
env -u GGML_HEXAGON_OPS650_FUSE -u GGML_HEXAGON_MOE_TOPK_ALIAS $L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-ppl4-s16def -LibDir $S16/lib -BinDir $S16/bin > npu_campaign/ops650-ppl4-s16def.out 2>&1

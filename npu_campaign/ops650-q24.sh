#!/bin/bash
# ops650 queue 24: FINAL package s16 (all gated fusions on by default, no env) plain full-vocab logits vs ref1 (merged2-m1, Hello, 128 rows); deletes the big logits files afterwards.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
P=ops650-g10
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-s16 -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-ops650-s16' -ExtraEnv "GGML_HEXAGON_MOE_TOPK_ALIAS=" -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin npu_campaign/mtp-speed/$P-s16.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp.txt 2>&1
echo "GATE g10 FINAL pkg s16 defaults plain logits vs merged2-m1 (ref1) exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-s16.logits.bin

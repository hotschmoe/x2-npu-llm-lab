#!/bin/bash
# ops650 queue 19: pkg s15 (= s14 with every gated fusion ON BY DEFAULT, no env) plain logits gate vs ref1 (merged2-m1, Hello 128 rows) + a default-config MTP? (no) 
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
P=ops650-g9
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-def -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-ops650-s15' -ExtraEnv "GGML_HEXAGON_MOE_TOPK_ALIAS=" -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin npu_campaign/mtp-speed/$P-def.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-def.txt 2>&1
echo "GATE g9 ref1 vs s15 defaults (fusions on by default, no env) exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-def.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-def.logits.bin

#!/bin/bash
# ops650 queue 16: NEGATIVE CONTROL for the vgather conv-window kernel (pkg s13, GGML_HEXAGON_OPS650_NEGCTRL=1): the plain-logit gate vs ref1 MUST FAIL.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
P=ops650-g7
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-neg -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-ops650-s13' -ExtraEnv "GGML_HEXAGON_OPS650_NEGCTRL=1" -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin npu_campaign/mtp-speed/$P-neg.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-neg.txt 2>&1
echo "GATE g7 NEGATIVE CONTROL (wrong conv tap plane) ref1 vs s13+NEGCTRL exit=$? (must be non-zero) :: $(tail -n 3 npu_campaign/mtp-speed/$P-cmp-neg.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-neg.logits.bin

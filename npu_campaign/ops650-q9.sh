#!/bin/bash
# ops650 queue 9: vectorised SSM_CONV+WIN (pkg s8 = s7 + gather/scatter conv window). profile op times + bit-exact gate.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-prof8 -Spec none -Tokens 24 -DeviceProfile 1 -Pkg 'npu_campaign\pkg-ops650-s8' -Prompt Hello > /dev/null 2>&1
P=ops650-g3
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-cv -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-ops650-s8' -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin npu_campaign/mtp-speed/$P-cv.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-ref1-cv.txt 2>&1
echo "GATE g3 ref1 vs fastdirty+vecconv(s8) exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-ref1-cv.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-cv.logits.bin

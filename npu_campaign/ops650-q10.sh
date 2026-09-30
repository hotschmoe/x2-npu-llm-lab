#!/bin/bash
# ops650 queue 10: all opt-in fusions on top of s9 (CONV_WIN_SILU 262144 | GDN_STATE 4096 | RMS_NORM_SCALE 1024 | SHEXP_EP 65536 | SILU_MUL 131072 + MOE_TOPK alias). Gate vs ref1, then per-fusion bisect if needed.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
P=ops650-g4
S9='npu_campaign\pkg-ops650-s9'
ALL="GGML_HEXAGON_OPS650_FUSE=463872;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-all -Spec none -Tokens 128 -LogitsTrace -Pkg "$S9" -ExtraEnv "$ALL" -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin npu_campaign/mtp-speed/$P-all.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-ref1-all.txt 2>&1
echo "GATE g4 ref1 vs s8+ALL optin fusions exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-ref1-all.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-all.logits.bin
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-prof-all -Spec none -Tokens 24 -DeviceProfile 1 -Pkg "$S9" -ExtraEnv "$ALL" -Prompt Hello > /dev/null 2>&1

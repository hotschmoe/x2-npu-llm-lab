#!/bin/bash
# ops650 queue 12 (pkg s12 = s11 + WINK vgather): MTP (draft-max 1, fused catch-up, dv 65536) device profile with all fusions on; then MTP target/draft logit gate
# (merged2-m1 vs s12 + all opt-in fusions).
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
ALL="GGML_HEXAGON_OPS650_FUSE=463872;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
S12='npu_campaign\pkg-ops650-s12'
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-mprof12-all -Spec mtp -DraftMax 1 -Tokens 32 -DeviceProfile 1 -Pkg "$S12" -ExtraEnv "$MT;$ALL" -Prompt Hello > /dev/null 2>&1
P=ops650-g5
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-refmtp -Spec mtp -DraftMax 1 -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-merged2-m1' -ExtraEnv "$MT" -Prompt Hello > /dev/null 2>&1
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-candmtp -Spec mtp -DraftMax 1 -Tokens 128 -LogitsTrace -Pkg "$S12" -ExtraEnv "$MT;$ALL" -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/$P-refmtp.logits.bin npu_campaign/mtp-speed/$P-candmtp.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-mtp-target.txt 2>&1
echo "GATE g5 MTP target rows merged2-m1 vs s12+ALL exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-mtp-target.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/$P-refmtp.logits.bin npu_campaign/mtp-speed/$P-candmtp.logits.bin --require-identical --context-index 1 --draft-greedy > npu_campaign/mtp-speed/$P-cmp-mtp-draft.txt 2>&1
echo "GATE g5 MTP draft rows exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-mtp-draft.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-refmtp.logits.bin npu_campaign/mtp-speed/$P-candmtp.logits.bin

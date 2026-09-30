#!/bin/bash
# ops650 queue 14 (pkg s14 = s12 + WINK SILU fold + MOE+TOPK for 2..4 rows): MTP profile and MTP target/draft logit gate vs merged2-m1, all opt-in fusions incl. bit 19.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
ALL="GGML_HEXAGON_OPS650_FUSE=988160;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
S14='npu_campaign\pkg-ops650-s14'
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-mprof14-all -Spec mtp -DraftMax 1 -Tokens 32 -DeviceProfile 1 -Pkg "$S14" -ExtraEnv "$MT;$ALL" -Prompt Hello > /dev/null 2>&1
P=ops650-g8
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-refmtp -Spec mtp -DraftMax 1 -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-merged2-m1' -ExtraEnv "$MT" -Prompt Hello > /dev/null 2>&1
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-candmtp -Spec mtp -DraftMax 1 -Tokens 128 -LogitsTrace -Pkg "$S14" -ExtraEnv "$MT;$ALL" -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/$P-refmtp.logits.bin npu_campaign/mtp-speed/$P-candmtp.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-mtp-target.txt 2>&1
echo "GATE g8 MTP target rows merged2-m1 vs s14+ALL(+TOPK_SMALLT) exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-mtp-target.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/$P-refmtp.logits.bin npu_campaign/mtp-speed/$P-candmtp.logits.bin --require-identical --context-index 1 --draft-greedy > npu_campaign/mtp-speed/$P-cmp-mtp-draft.txt 2>&1
echo "GATE g8 MTP draft rows exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-mtp-draft.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-refmtp.logits.bin npu_campaign/mtp-speed/$P-candmtp.logits.bin

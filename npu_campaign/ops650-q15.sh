#!/bin/bash
# ops650 queue 15: correctness battery for pkg s12 (+ALL opt-in fusions): PPL 4x512 and 16 chunks (expect 12.1307 / 13.3331), default-config PPL,
# fusion edge check, and plain full-vocab logits on two more prompts vs merged2-m1.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
S12=npu_campaign/pkg-ops650-s12
export GGML_HEXAGON_OPS650_FUSE=463872 GGML_HEXAGON_MOE_TOPK_ALIAS=1
$L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-s12all-4x512 -LibDir $S12/lib -BinDir $S12/bin > npu_campaign/ops650-ppl-s12all-4x512.out 2>&1
$L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-s12all-16 -LibDir $S12/lib -BinDir $S12/bin -Chunks 16 > npu_campaign/ops650-ppl-s12all-16.out 2>&1
unset GGML_HEXAGON_OPS650_FUSE GGML_HEXAGON_MOE_TOPK_ALIAS
$L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-s12def-4x512 -LibDir $S12/lib -BinDir $S12/bin > npu_campaign/ops650-ppl-s12def-4x512.out 2>&1
export GGML_HEXAGON_OPS650_FUSE=463872 GGML_HEXAGON_MOE_TOPK_ALIAS=1
$L powershell -NoProfile -File npu_campaign/run-fusion-edge-check.ps1 -LibDir $S12/lib -BinDir $S12/bin -Tag ops650edge > npu_campaign/ops650-edge-s12all.out 2>&1
unset GGML_HEXAGON_OPS650_FUSE GGML_HEXAGON_MOE_TOPK_ALIAS
ALL="GGML_HEXAGON_OPS650_FUSE=463872;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
P=ops650-g6
i=0
for pr in "Explain why the sky appears blue in three clear sentences." "Explain how a hash table handles collisions, with a concrete example."; do
  i=$((i+1))
  $L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-p$i-ref -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-merged2-m1' -Prompt "$pr" > /dev/null 2>&1
  $L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-p$i-cand -Spec none -Tokens 128 -LogitsTrace -Pkg "$S12" -ExtraEnv "$ALL" -Prompt "$pr" > /dev/null 2>&1
  python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/$P-p$i-ref.logits.bin npu_campaign/mtp-speed/$P-p$i-cand.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-p$i.txt 2>&1
  echo "GATE g6 prompt$i plain ref(merged2-m1) vs s12+ALL exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-p$i.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
  rm -f npu_campaign/mtp-speed/$P-p$i-ref.logits.bin npu_campaign/mtp-speed/$P-p$i-cand.logits.bin
done

#!/bin/bash
# ops650 master queue 18 (sequential, priority order). pkg s14; ALL = every opt-in fusion (mask 988160 incl. small-T router top-k) + MOE_TOPK alias.
# 1) plain decode ABBA base vs s14F (3 rounds)  2) correctness battery (PPL x2, edge check, extra prompt logits)  3) MTP ABBA (3 rounds)  4) host knobs (OPBATCH 1280, spin)
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
S14=npu_campaign/pkg-ops650-s14
ALL="GGML_HEXAGON_OPS650_FUSE=988160;GGML_HEXAGON_MOE_TOPK_ALIAS=1"
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
run() { # label spec pkg extraenv [extra args...]
  local label=$1 spec=$2 pkg=$3 ev=$4; shift 4
  $L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label "$label" -Spec $spec -Tokens 128 -Pkg "$pkg" -ExtraEnv "$ev" -Prompt Hello "$@" > /dev/null 2>&1
}
# 1) plain ABBA (ab2 base-r1 already exists from the first queue: reuse as round-1 base)
run ops650-ab2-s14F-r1 none "$S14" "$ALL"
run ops650-ab2-s14F-r2 none "$S14" "$ALL"
run ops650-ab2-base-r2 none 'npu_campaign\pkg-merged2-m1' "X=1"
run ops650-ab2-base-r3 none 'npu_campaign\pkg-merged2-m1' "X=1"
run ops650-ab2-s14F-r3 none "$S14" "$ALL"
# 2) correctness battery
export GGML_HEXAGON_OPS650_FUSE=988160 GGML_HEXAGON_MOE_TOPK_ALIAS=1
$L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-s14all-4x512 -LibDir $S14/lib -BinDir $S14/bin > npu_campaign/ops650-ppl-s14all-4x512.out 2>&1
$L powershell -NoProfile -File npu_campaign/run-prefill-ppl.ps1 -Label ops650-s14all-16 -LibDir $S14/lib -BinDir $S14/bin -Chunks 16 > npu_campaign/ops650-ppl-s14all-16.out 2>&1
$L powershell -NoProfile -File npu_campaign/run-fusion-edge-check.ps1 -LibDir $S14/lib -BinDir $S14/bin -Tag ops650edge > npu_campaign/ops650-edge-s14all.out 2>&1
unset GGML_HEXAGON_OPS650_FUSE GGML_HEXAGON_MOE_TOPK_ALIAS
P=ops650-g6
pr="Explain how a hash table handles collisions, with a concrete example."
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-ref -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-merged2-m1' -Prompt "$pr" > /dev/null 2>&1
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-cand -Spec none -Tokens 128 -LogitsTrace -Pkg "$S14" -ExtraEnv "$ALL" -Prompt "$pr" > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/$P-ref.logits.bin npu_campaign/mtp-speed/$P-cand.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp.txt 2>&1
echo "GATE g6 hash-table prompt plain ref(merged2-m1) vs s14+ALL exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-ref.logits.bin npu_campaign/mtp-speed/$P-cand.logits.bin
# 3) MTP ABBA
run ops650-abm-base-r1 mtp 'npu_campaign\pkg-merged2-m1' "$MT" -DraftMax 1
run ops650-abm-s14F-r1 mtp "$S14" "$MT;$ALL" -DraftMax 1
run ops650-abm-s14F-r2 mtp "$S14" "$MT;$ALL" -DraftMax 1
run ops650-abm-base-r2 mtp 'npu_campaign\pkg-merged2-m1' "$MT" -DraftMax 1
run ops650-abm-base-r3 mtp 'npu_campaign\pkg-merged2-m1' "$MT" -DraftMax 1
run ops650-abm-s14F-r3 mtp "$S14" "$MT;$ALL" -DraftMax 1
# 4) host knobs (fusions off, timing envs on)
T="LLAMA_DECODE_TIMING=1;GGML_HEXAGON_TIME_BATCH=1"
run ops650-h2-A_base-r1 none "$S14" "$T;X=1"
run ops650-h2-B_ob1280-r1 none "$S14" "$T;GGML_HEXAGON_OPBATCH=1280"
run ops650-h2-D_spin-r1 none "$S14" "$T;GGML_HEXAGON_OPPOLL=1"

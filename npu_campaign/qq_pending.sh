#!/bin/bash
# PENDING RUNS (speed A/B, item D head speed, item B garbage hunt): run on AC power only. Fresh labels (d2s2/d2h2/d2b2). PPL for the Q4_0 head is section 4.
cd /c/Users/hotschmoe/documents/llm_inference
LR="python npu_campaign/lockrun.py --"
PS="powershell -NoProfile -File"
tps() { grep -o '[0-9.]* tokens per second' npu_campaign/mtp-speed/$1.detail.log | tail -1; }
acc() { grep -o 'draft acceptance[^)]*)' npu_campaign/mtp-speed/$1.detail.log | tail -1; }
FCDV='LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536'
echo "== 1. speed ABBA (Hello 256): plain base/fix64, MTP base/fix64"
n=0
for order in "base fix64" "fix64 base" "base fix64"; do
  for v in $order; do n=$((n+1))
    $LR $PS npu_campaign/run-mtp-speed.ps1 -Label d2s2-plain-$v-$n -Pkg npu_campaign/pkg-decode2-$v -Spec none -Tokens 256 -Prompt Hello > /dev/null 2>&1
    echo "plain $v $n $(tps d2s2-plain-$v-$n)"
  done
done
for order in "base fix64" "fix64 base"; do
  for v in $order; do n=$((n+1))
    $LR $PS npu_campaign/run-mtp-speed.ps1 -Label d2s2-mtp-$v-$n -Pkg npu_campaign/pkg-decode2-$v -Spec mtp -DraftMax 1 -Tokens 256 -Prompt Hello -ExtraEnv "$FCDV" > /dev/null 2>&1
    echo "mtp $v $n $(tps d2s2-mtp-$v-$n) $(acc d2s2-mtp-$v-$n)"
  done
done
echo "== 4. item D head Q4_0"
M6='models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf'
M4='models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32-headq40.gguf'
$LR $PS npu_campaign/run-prefill-ppl.ps1 -Label d2head-q6 -LibDir npu_campaign/pkg-decode2-fix64/lib -BinDir npu_campaign/pkg-decode2-fix64/bin -Model "$M6" 2>&1 | tail -2
$LR $PS npu_campaign/run-prefill-ppl.ps1 -Label d2head-q40 -LibDir npu_campaign/pkg-decode2-fix64/lib -BinDir npu_campaign/pkg-decode2-fix64/bin -Model "$M4" 2>&1 | tail -2
i=0
for order in "6 4" "4 6" "6 4"; do
  for h in $order; do i=$((i+1)); if [ $h = 6 ]; then M=$M6; else M=$M4; fi
    $LR $PS npu_campaign/run-mtp-speed.ps1 -Label d2h2-plain-h$h-$i -Pkg npu_campaign/pkg-decode2-fix64 -Spec none -Tokens 256 -Prompt Hello -ModelPath "$M" > /dev/null 2>&1
    echo "plain head=$h run $i $(tps d2h2-plain-h$h-$i)"
  done
done
for order in "6 4" "4 6"; do
  for h in $order; do i=$((i+1)); if [ $h = 6 ]; then M=$M6; else M=$M4; fi
    $LR $PS npu_campaign/run-mtp-speed.ps1 -Label d2h2-mtp-h$h-$i -Pkg npu_campaign/pkg-decode2-fix64 -Spec mtp -DraftMax 1 -Tokens 256 -Prompt Hello -ModelPath "$M" -ExtraEnv "$FCDV" > /dev/null 2>&1
    echo "mtp head=$h run $i $(tps d2h2-mtp-h$h-$i) $(acc d2h2-mtp-h$h-$i)"
  done
done
echo "== 5. item B garbage hunt (slots=4)"
SKY='Explain why the sky appears blue in three clear sentences.'
for i in 1 2 3 4; do for v in base fix64; do
  $LR $PS npu_campaign/run-mtp-speed.ps1 -Label d2b2-$v-d1p7-$i -Pkg npu_campaign/pkg-decode2-$v -Spec mtp -DraftMax 1 -PMin 0.7 -Tokens 256 -Prompt Hello -ExtraEnv "$FCDV;GGML_HEXAGON_REPLAY_SLOTS=4" > /dev/null 2>&1
  echo "d1p7 $v $i slashes=$(grep -c '////' npu_campaign/mtp-speed/d2b2-$v-d1p7-$i.log) $(tps d2b2-$v-d1p7-$i) $(acc d2b2-$v-d1p7-$i)"
done; done
for i in 1 2 3 4; do for v in base fix64; do
  $LR $PS npu_campaign/run-mtp-speed.ps1 -Label d2b2-$v-d2p7-$i -Pkg npu_campaign/pkg-decode2-$v -Spec mtp -DraftMax 2 -PMin 0.7 -Tokens 256 -Prompt "$SKY" -ExtraEnv "LLAMA_MTP_FUSED_CATCHUP=1;GGML_HEXAGON_REPLAY_SLOTS=4" > /dev/null 2>&1
  echo "d2p7 $v $i slashes=$(grep -c '////' npu_campaign/mtp-speed/d2b2-$v-d2p7-$i.log) $(tps d2b2-$v-d2p7-$i) $(acc d2b2-$v-d2p7-$i)"
done; done
echo "== done"

#!/bin/bash
# q6: OPPOLL=1 spin stress with the bounded-spin/watchdog build (pkg-qwen4-s14): N runs, alternating plain / MTP d1, 48 tokens; the run script kills a run after 150 s.
# Hold the NPU lock for 5 runs at a time (lockrun.py) so others can interleave. Result lines -> qwen4-spin-stress.out
cd /c/Users/hotschmoe/documents/llm_inference
P='npu_campaign\pkg-qwen4-s14'
N=${1:-30}
i=0
while [ $i -lt $N ]; do
  python npu_campaign/lockrun.py -- 'C:\Program Files\Git\usr\bin\bash.exe' -c '
    cd /c/Users/hotschmoe/documents/llm_inference
    for j in 1 2 3 4; do
      k=$(( '$i' + j ))
      if [ $(( k % 2 )) = 1 ]; then SPEC=none; X="GGML_HEXAGON_OPPOLL=1"; else SPEC=mtp; X="GGML_HEXAGON_OPPOLL=1;LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"; fi
      L=q4-spin-$k
      python npu_campaign/qwen4-guard.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $L -Spec $SPEC -DraftMax 1 -Tokens 48 -Pkg "'"$P"'" -ExtraEnv "$X" -TimeoutSec 150 > /dev/null 2>&1
      rc=$?
      tps=$(grep -o "[0-9.]* tokens per second" npu_campaign/mtp-speed/$L.detail.log 2>/dev/null | tail -1 | cut -d" " -f1)
      resc=$(grep -c "OPPOLL spin gave up" npu_campaign/mtp-speed/$L.detail.log 2>/dev/null)
      echo "$L spec=$SPEC rc=$rc tps=$tps rescues=$resc" >> npu_campaign/qwen4-spin-stress.out
    done
  '
  i=$(( i + 4 ))
done

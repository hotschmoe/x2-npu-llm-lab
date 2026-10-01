#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
# wait until master1 finished (keeps the lock queue orderly)
while ! grep -q "== done" npu_campaign/m3_master1.out 2>/dev/null; do sleep 30; done
SLOTS='GGML_HEXAGON_REPLAY_SLOTS=4'
BASE='npu_campaign\pkg-decode2-base'
echo "== I. garbage hunt: NOT-recommended configs. installed(fix) vs decode2-base (pre-flush-fix control)"
for i in 1 2 3 4 5 6; do
  for v in wos base; do
    if [ $v = wos ]; then P="$PKG_WOS"; else P="$BASE"; fi
    run m3i-d1p7-$v-$i "$P" mtp 1 256 Hello "$FCDV;$SLOTS" "" 0.7
    echo "   slashes=$(grep -c '////' $D/m3i-d1p7-$v-$i.log)"
    run m3i-d2p7-$v-$i "$P" mtp 2 256 "$SKY" "LLAMA_MTP_FUSED_CATCHUP=1;$SLOTS" "" 0.7
    echo "   slashes=$(grep -c '////' $D/m3i-d2p7-$v-$i.log)"
  done
done
echo "== J. draft-max 2 / 3 (installed): Hello and sky, 2 rounds, plain fc-off (auto) config, no p-min"
n=0
for pr in Hello sky; do
  if [ $pr = sky ]; then T="$SKY"; else T=Hello; fi
  for ord in "1 2 3" "3 2 1"; do for dm in $ord; do n=$((n+1))
    run m3j-$pr-d$dm-$n "$PKG_WOS" mtp $dm 256 "$T" "$FCDV"
  done; done
done
echo "== done"

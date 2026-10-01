#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PB='npu_campaign\pkg-mtp3-b'
echo "== G. fastrpc QoS variants (plain decode, Hello256), 2 rounds, pkg-mtp3-b"
n=0
for ord in "def q1_100 q2_100 q3_10000 spin" "spin q3_10000 q2_100 q1_100 def"; do for v in $ord; do n=$((n+1))
  case $v in def) E="" ;; q1_100) E="GGML_HEXAGON_QOS=1,100" ;; q2_100) E="GGML_HEXAGON_QOS=2,100" ;; q3_10000) E="GGML_HEXAGON_QOS=3,10000" ;; spin) E="GGML_HEXAGON_OPPOLL=1" ;; esac
  run m3g-$v-$n "$PB" none 1 256 Hello "$E"
done; done
echo "== done"

#!/bin/bash
# layout-abba.sh TAG PKG VAR ROUNDS : ABBA interleave of VAR unset / VAR=1 on one package
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign
TAG=$1; PK=$2; VAR=$3; R=${4:-3}
for i in $(seq 1 $R); do
  if [ $((i%2)) = 1 ]; then ./layout-pp.sh $TAG-off-r$i "$PK" ""; ./layout-pp.sh $TAG-on-r$i "$PK" "$VAR=1"
  else ./layout-pp.sh $TAG-on-r$i "$PK" "$VAR=1"; ./layout-pp.sh $TAG-off-r$i "$PK" ""; fi
done

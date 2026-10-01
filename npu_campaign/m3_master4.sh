#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
echo "== H. target backend sampling (-bs): plain/mtp, greedy Hello256 and temp 0.6 (code prompt), 2 rounds"
T06="--temp 0.6 --top-p 0.95 --top-k 20"
n=0
for ord in "plain plainbs mtp mtpbs" "mtpbs mtp plainbs plain"; do for v in $ord; do n=$((n+1))
  case $v in plain) run m3h-g-$v-$n "$PKG_WOS" none 1 256 Hello "" ;;
    plainbs) run m3h-g-$v-$n "$PKG_WOS" none 1 256 Hello "" "--seed 1234 --backend-sampling" ;;
    mtp) run m3h-g-$v-$n "$PKG_WOS" mtp 1 256 Hello "$FCDV" ;;
    mtpbs) run m3h-g-$v-$n "$PKG_WOS" mtp 1 256 Hello "$FCDV" "--seed 1234 --backend-sampling" ;; esac
done; done
n=0
for ord in "mtp mtpbs plain plainbs" "plainbs plain mtpbs mtp"; do for v in $ord; do n=$((n+1))
  case $v in plain) run m3h-t-$v-$n "$PKG_WOS" none 1 256 "$CODE" "" "$T06" ;;
    plainbs) run m3h-t-$v-$n "$PKG_WOS" none 1 256 "$CODE" "" "$T06 --backend-sampling" ;;
    mtp) run m3h-t-$v-$n "$PKG_WOS" mtp 1 256 "$CODE" "$FCDV" "$T06" ;;
    mtpbs) run m3h-t-$v-$n "$PKG_WOS" mtp 1 256 "$CODE" "$FCDV" "$T06 --backend-sampling" ;; esac
done; done
echo "== done"

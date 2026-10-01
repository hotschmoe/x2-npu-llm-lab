#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
echo "== profiles"
$LR $PSF npu_campaign/run-mtp-speed.ps1 -Label m3p-plain -Spec none -Tokens 24 -DeviceProfile 1 >/dev/null 2>&1
$LR $PSF npu_campaign/run-mtp-speed.ps1 -Label m3p-mtp -Spec mtp -DraftMax 1 -Tokens 24 -DeviceProfile 1 -ExtraEnv "$FCDV" -MtpProfile >/dev/null 2>&1
echo "profiles done"
echo "== A. synthetic acceptance ceiling (SPOOF, output invalid), dmax1, block poll, Hello256"
for r in 1.0 0.85 0.5 0.0; do run m3a-synth$r "$PKG_WOS" mtp 1 256 Hello "$FCDV" "--spec-synth-rates $r"; done
echo "== B. baseline ABBA Hello256 x3: plain, mtp, plainspin, mtpspin"
n=0
for round in 1 2 3; do
  if [ $((round%2)) = 1 ]; then ord="plain mtp plainspin mtpspin"; else ord="mtpspin plainspin mtp plain"; fi
  for v in $ord; do n=$((n+1))
    case $v in plain) run m3b-$v-$n "$PKG_WOS" none 1 256 Hello "" ;;
      mtp) run m3b-$v-$n "$PKG_WOS" mtp 1 256 Hello "$FCDV" ;;
      plainspin) run m3b-$v-$n "$PKG_WOS" none 1 256 Hello "GGML_HEXAGON_OPPOLL=1" ;;
      mtpspin) run m3b-$v-$n "$PKG_WOS" mtp 1 256 Hello "$FCDV;GGML_HEXAGON_OPPOLL=1" ;; esac
  done
done
echo "== C. agentic prompts greedy plain vs mtp (2 rounds each)"
n=0
for pr in code json sky; do
  case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; esac
  for ord in "none mtp" "mtp none"; do for s in $ord; do n=$((n+1))
    if [ $s = mtp ]; then E="$FCDV"; else E=""; fi
    run m3c-g-$pr-$s-$n "$PKG_WOS" $s 1 256 "$T" "$E"
  done; done
done
echo "== D. temp 0.6 top-p 0.95 top-k 20"
n=0
for pr in code json sky; do
  case $pr in code) T="$CODE";; json) T="$JSON";; sky) T="$SKY";; esac
  for ord in "none mtp" "mtp none"; do for s in $ord; do n=$((n+1))
    if [ $s = mtp ]; then E="$FCDV"; else E=""; fi
    run m3d-t-$pr-$s-$n "$PKG_WOS" $s 1 256 "$T" "$E" "--temp 0.6 --top-p 0.95 --top-k 20"
  done; done
done
echo "== done"

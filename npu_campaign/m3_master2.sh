#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PA='npu_campaign\pkg-mtp3-a'
REP='Here is a config: {host: localhost, port: 8080, timeout: 30, retries: 3, workers: 4}. Copy it exactly three times, changing only the port each time (8081, 8082, 8083).'
echo "== E. ngram+MTP (size-n 4, size-m 2), needs n_rs_seq 2; greedy; plain vs mtp vs mtp+ngram"
n=0
for pr in rep code; do
  case $pr in rep) T="$REP";; code) T="$CODE";; esac
  for ord in "plain mtp mng" "mng mtp plain"; do for v in $ord; do n=$((n+1))
    case $v in plain) run m3e-$pr-$v-$n "$PA" none 1 256 "$T" "" ;;
      mtp) run m3e-$pr-$v-$n "$PA" mtp 1 256 "$T" "$FCDV" ;;
      mng) run m3e-$pr-$v-$n "$PA" mtpngram 1 256 "$T" "$FCDV" "--spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 2" ;; esac
  done; done
done
echo "== F. Q4_0 head: PPL then speed ABBA (Hello 256), block poll"
M6='models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf'
M4='models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32-headq40.gguf'
for h in 6 4; do if [ $h = 6 ]; then M=$M6; else M=$M4; fi
  $LR $PSF npu_campaign/run-prefill-ppl.ps1 -Label m3f-ppl-q$h -LibDir llama.cpp-hexagon/pkg-wos/lib -BinDir llama.cpp-hexagon/pkg-wos/bin -Model "$M" 2>&1 | tail -2
done
i=0
for ord in "6 4" "4 6" "6 4"; do for h in $ord; do i=$((i+1)); if [ $h = 6 ]; then M=$M6; else M=$M4; fi
  $LR $PSF npu_campaign/run-mtp-speed.ps1 -Label m3f-plain-h$h-$i -Pkg "$PKG_WOS" -Spec none -Tokens 256 -Prompt Hello -ModelPath "$M" >/dev/null 2>&1
  echo "plain head=$h $i $(tps m3f-plain-h$h-$i)"
done; done
for ord in "6 4" "4 6" "6 4"; do for h in $ord; do i=$((i+1)); if [ $h = 6 ]; then M=$M6; else M=$M4; fi
  $LR $PSF npu_campaign/run-mtp-speed.ps1 -Label m3f-mtp-h$h-$i -Pkg "$PKG_WOS" -Spec mtp -DraftMax 1 -Tokens 256 -Prompt Hello -ModelPath "$M" -ExtraEnv "$FCDV" >/dev/null 2>&1
  echo "mtp head=$h $i $(tps m3f-mtp-h$h-$i) $(acc m3f-mtp-h$h-$i)"
done; done
echo "== done"

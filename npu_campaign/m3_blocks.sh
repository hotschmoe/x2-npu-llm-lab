#!/bin/bash
# usage: m3_blocks.sh block NAME   (run inside the NPU lock)   |   m3_blocks.sh all NAME1 NAME2 ...  (driver: each block under lockrun3)
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
if [ "$1" = all ]; then shift; for b in "$@"; do echo "## block $b"; $LR "C:/Program Files/Git/usr/bin/bash.exe" npu_campaign/m3_blocks.sh block $b; done; echo "== all done"; exit 0; fi
b=$2
case $b in
 qos1|qos2) # fastrpc QoS variants, plain decode
   if [ $b = qos1 ]; then ord="def q1_100 q2_100 q3_10000 spin"; else ord="spin q3_10000 q2_100 q1_100 def"; fi
   for v in $ord; do case $v in def) E="" ;; q1_100) E="GGML_HEXAGON_QOS=1,100" ;; q2_100) E="GGML_HEXAGON_QOS=2,100" ;; q3_10000) E="GGML_HEXAGON_QOS=3,10000" ;; spin) E="GGML_HEXAGON_OPPOLL=1" ;; esac
     run m3g-$b-$v "$PB" none 1 256 Hello "$E"; done ;;
 base1|base2|base3) # baseline ABBA on installed
   case $b in base1|base3) ord="plain mtp plainspin mtpspin";; base2) ord="mtpspin plainspin mtp plain";; esac
   for v in $ord; do case $v in plain) run m3b-$b-$v "$PKG_WOS" none 1 256 Hello "" ;; mtp) run m3b-$b-$v "$PKG_WOS" mtp 1 256 Hello "$FCDV" ;;
     plainspin) run m3b-$b-$v "$PKG_WOS" none 1 256 Hello "GGML_HEXAGON_OPPOLL=1" ;; mtpspin) run m3b-$b-$v "$PKG_WOS" mtp 1 256 Hello "$FCDV;GGML_HEXAGON_OPPOLL=1" ;; esac; done ;;
 ng-rep1|ng-rep2|ng-code1) # ngram+mtp
   case $b in ng-rep1) T="$REP"; ord="plain mtp mng";; ng-rep2) T="$REP"; ord="mng mtp plain";; ng-code1) T="$CODE"; ord="mng mtp plain";; esac
   for v in $ord; do case $v in plain) run m3e-$b-$v "$PA" none 1 256 "$T" "" ;; mtp) run m3e-$b-$v "$PA" mtp 1 256 "$T" "$FCDV" ;;
     mng) run m3e-$b-$v "$PA" mtpngram 1 256 "$T" "$FCDV" "--spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 2" ;; esac; done ;;
 ppl6) $PSF npu_campaign/run-prefill-ppl.ps1 -Label m3f-ppl-q6 -LibDir llama.cpp-hexagon/pkg-wos/lib -BinDir llama.cpp-hexagon/pkg-wos/bin -Model "$M6" 2>&1 | tail -2 ;;
 ppl4) $PSF npu_campaign/run-prefill-ppl.ps1 -Label m3f-ppl-q4 -LibDir llama.cpp-hexagon/pkg-wos/lib -BinDir llama.cpp-hexagon/pkg-wos/bin -Model "$M4" 2>&1 | tail -2 ;;
 head1|head2|head3) # Q4_0 head vs Q6_K head, mtp + plain
   case $b in head1|head3) ord="6 4";; head2) ord="4 6";; esac
   for h in $ord; do if [ $h = 6 ]; then M=$M6; else M=$M4; fi
     $PSF npu_campaign/run-mtp-speed.ps1 -Label m3f-mtp-$b-h$h -Pkg "$PKG_WOS" -Spec mtp -DraftMax 1 -Tokens 256 -Prompt Hello -ModelPath "$M" -ExtraEnv "$FCDV" >/dev/null 2>&1; echo "mtp head=$h $b $(tps m3f-mtp-$b-h$h) $(acc m3f-mtp-$b-h$h)"
     $PSF npu_campaign/run-mtp-speed.ps1 -Label m3f-plain-$b-h$h -Pkg "$PKG_WOS" -Spec none -Tokens 256 -Prompt Hello -ModelPath "$M" >/dev/null 2>&1; echo "plain head=$h $b $(tps m3f-plain-$b-h$h)"; done ;;
 bs1|bs2) # backend sampling greedy
   if [ $b = bs1 ]; then ord="plain plainbs mtp mtpbs"; else ord="mtpbs mtp plainbs plain"; fi
   for v in $ord; do case $v in plain) run m3h-$b-$v "$PKG_WOS" none 1 256 Hello "" ;; plainbs) run m3h-$b-$v "$PKG_WOS" none 1 256 Hello "" "--seed 1234 --backend-sampling" ;;
     mtp) run m3h-$b-$v "$PKG_WOS" mtp 1 256 Hello "$FCDV" ;; mtpbs) run m3h-$b-$v "$PKG_WOS" mtp 1 256 Hello "$FCDV" "--seed 1234 --backend-sampling" ;; esac; done ;;
 bst1|bst2) # backend sampling at temp 0.6 (code)
   if [ $b = bst1 ]; then ord="plain plainbs mtp mtpbs"; else ord="mtpbs mtp plainbs plain"; fi
   for v in $ord; do case $v in plain) run m3h-$b-$v "$PKG_WOS" none 1 256 "$CODE" "" "$T06" ;; plainbs) run m3h-$b-$v "$PKG_WOS" none 1 256 "$CODE" "" "$T06 --backend-sampling" ;;
     mtp) run m3h-$b-$v "$PKG_WOS" mtp 1 256 "$CODE" "$FCDV" "$T06" ;; mtpbs) run m3h-$b-$v "$PKG_WOS" mtp 1 256 "$CODE" "$FCDV" "$T06 --backend-sampling" ;; esac; done ;;
 ag1|ag2|ag3) # agentic-style prompts greedy: plain vs mtp for code/json/sky
   case $b in ag1) T="$CODE";; ag2) T="$JSON";; ag3) T="$SKY";; esac
   run m3c-$b-plain "$PKG_WOS" none 1 256 "$T" ""; run m3c-$b-mtp "$PKG_WOS" mtp 1 256 "$T" "$FCDV"
   run m3c-$b-mtp2 "$PKG_WOS" mtp 1 256 "$T" "$FCDV"; run m3c-$b-plain2 "$PKG_WOS" none 1 256 "$T" "" ;;
 tp1|tp2|tp3) # temp 0.6 acceptance: plain vs mtp
   case $b in tp1) T="$CODE";; tp2) T="$JSON";; tp3) T="$SKY";; esac
   run m3d-$b-plain "$PKG_WOS" none 1 256 "$T" "" "$T06"; run m3d-$b-mtp "$PKG_WOS" mtp 1 256 "$T" "$FCDV" "$T06"
   run m3d-$b-mtp2 "$PKG_WOS" mtp 1 256 "$T" "$FCDV" "$T06"; run m3d-$b-plain2 "$PKG_WOS" none 1 256 "$T" "" "$T06" ;;
 gh-wos-d1|gh-base-d1|gh-wos-d2|gh-base-d2) # garbage hunt: 3 runs per block
   case $b in gh-wos-*) P="$PKG_WOS";; gh-base-*) P='npu_campaign\pkg-decode2-base';; esac
   for i in 1 2 3; do
     if [[ $b == *d1 ]]; then run m3i-$b-$i "$P" mtp 1 256 Hello "$FCDV;GGML_HEXAGON_REPLAY_SLOTS=4" "" 0.7
     else run m3i-$b-$i "$P" mtp 2 256 "$SKY" "LLAMA_MTP_FUSED_CATCHUP=1;GGML_HEXAGON_REPLAY_SLOTS=4" "" 0.7; fi
     echo "   slashes=$(grep -c '////' $D/m3i-$b-$i.log)"; done ;;
 dm1|dm2) # draft-max sweep
   if [ $b = dm1 ]; then ord="1 2 3"; else ord="3 2 1"; fi
   for dm in $ord; do run m3j-$b-hello-d$dm "$PKG_WOS" mtp $dm 256 Hello "$FCDV"; done
   for dm in $ord; do run m3j-$b-sky-d$dm "$PKG_WOS" mtp $dm 256 "$SKY" "$FCDV"; done ;;
 dmrep1|dmrep2) # draft-max sweep on a copy-heavy prompt (high acceptance)
   if [ $b = dmrep1 ]; then ord="1 2 3"; else ord="3 2 1"; fi
   for dm in $ord; do run m3k-$b-rep-d$dm "$PKG_WOS" mtp $dm 256 "$REP" "$FCDV"; done
   for dm in $ord; do run m3k-$b-json-d$dm "$PKG_WOS" mtp $dm 256 "$JSON" "$FCDV"; done ;;
 dmp1|dmp2) # dmax 2 gated by draft-head confidence for the 2nd token (LLAMA_MTP_PMIN_NEXT), vs dmax 1
   if [ $b = dmp1 ]; then ord="d1 d2n8 d2n9 d2n0"; else ord="d2n0 d2n9 d2n8 d1"; fi
   for pr in rep hello json; do case $pr in rep) T="$REP";; hello) T=Hello;; json) T="$JSON";; esac
     for v in $ord; do case $v in d1) run m3l-$b-$pr-$v "$PKG_WOS" mtp 1 256 "$T" "$FCDV" ;;
       d2n8) run m3l-$b-$pr-$v "$PKG_WOS" mtp 2 256 "$T" "LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" ;;
       d2n9) run m3l-$b-$pr-$v "$PKG_WOS" mtp 2 256 "$T" "LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.9" ;;
       d2n0) run m3l-$b-$pr-$v "$PKG_WOS" mtp 2 256 "$T" "LLAMA_MTP_DRAFT_VOCAB=65536" ;; esac; done; done ;;
 fcd1|fcd2) # forced fused catch-up with draft-max 2 (pkg d) vs unfused, gated 0.8
   PD='npu_campaign\pkg-mtp3-d'
   if [ $b = fcd1 ]; then ord="nofc fc"; else ord="fc nofc"; fi
   for pr in rep json hello; do case $pr in rep) T="$REP";; hello) T=Hello;; json) T="$JSON";; esac
     for v in $ord; do case $v in nofc) run m3m-$b-$pr-$v "$PD" mtp 2 256 "$T" "LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" ;;
       fc) run m3m-$b-$pr-$v "$PD" mtp 2 256 "$T" "LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_FC_FORCE=1;LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" ;; esac; done; done ;;
 *) echo "unknown block $b" ;;
esac

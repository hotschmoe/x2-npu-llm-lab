#!/bin/bash
# ops650 queue 8: fast dirty tracker (pkg s7). 1) profile seg run, 2) bit-exact plain logits vs ref1, 3) interleaved untraced ABBA vs installed merged2-m1.
cd /c/Users/hotschmoe/documents/llm_inference
L='python npu_campaign/lockrun650.py --'
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-seg7 -Spec none -Tokens 24 -DeviceProfile 1 -Pkg 'npu_campaign\pkg-ops650-s7' -ExtraEnv "GGML_HEXAGON_OPS650_SPOOF_CYCLE=0,1,0" -Prompt Hello
P=ops650-g2
$L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $P-fd -Spec none -Tokens 128 -LogitsTrace -Pkg 'npu_campaign\pkg-ops650-s7' -Prompt Hello > /dev/null 2>&1
python npu_campaign/compare_logits_traces.py npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin npu_campaign/mtp-speed/$P-fd.logits.bin --require-identical > npu_campaign/mtp-speed/$P-cmp-ref1-fd.txt 2>&1
echo "GATE g2 ref1 vs fastdirty(s7) exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-ref1-fd.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out
rm -f npu_campaign/mtp-speed/$P-fd.logits.bin
for r in 1 2 3; do
  if [ $((r%2)) -eq 1 ]; then O="base fd"; else O="fd base"; fi
  for k in $O; do
    if [ $k = base ]; then PK='npu_campaign\pkg-merged2-m1'; else PK='npu_campaign\pkg-ops650-s7'; fi
    $L powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label ops650-ab1-$k-r$r -Spec none -Tokens 128 -Pkg "$PK" -Prompt Hello > /dev/null 2>&1
  done
done

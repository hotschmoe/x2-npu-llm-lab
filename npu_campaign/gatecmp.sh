#!/bin/bash
# compare the four traces of an mtp-gate prefix once complete (the gate script aborts before deleting them), then delete them
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign/mtp-speed
for k in sky hash; do
  while [ ! -f d2g-$k-cand-mtp.detail.log ] || ! grep -q "draft acceptance" d2g-$k-cand-mtp.detail.log 2>/dev/null || [ -f ../.npu.lock/owner.txt -a -n "$(grep d2g-$k-cand-mtp ../.npu.lock/owner.txt 2>/dev/null)" ]; do sleep 10; done
  sleep 20
  for pair in "ref-plain cand-plain" "ref-mtp cand-mtp" "cand-plain cand-mtp" "ref-plain ref-mtp"; do
    set -- $pair
    r=$(python ../compare_logits_traces.py d2g-$k-$1.logits.bin d2g-$k-$2.logits.bin --require-identical 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-300)
    echo "GATEMAN $k $1 vs $2 :: $r"
  done
  grep -o "draft acceptance[^)]*)" d2g-$k-ref-mtp.detail.log | tail -1; grep -o "draft acceptance[^)]*)" d2g-$k-cand-mtp.detail.log | tail -1
  rm -f d2g-$k-*.logits.bin
done

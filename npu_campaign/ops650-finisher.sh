#!/bin/bash
# ops650 finisher: waits for the tail of the master queue (ops650-q18.sh: MTP ABBA + host knobs) and summarises the results into
# npu_campaign/ops650-final-results.txt and an appended section of OPS650_LOG.md. Pure file watching, takes no NPU lock.
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign
until grep -q "eval time" mtp-speed/ops650-h2-D_spin-r1.detail.log 2>/dev/null; do sleep 60; done
{
  echo "=== plain decode ABBA (ops650-ab2)"; python ops650_ab.py ops650-ab2
  echo; echo "=== MTP ABBA (ops650-abm): draft-max 1, fused catch-up, dv 65536, Hello 128 tokens"; python ops650_ab.py ops650-abm
  echo; echo "=== host knobs (ops650-h2, pkg s14 fusions off, 1 round): A_base OPBATCH 3072 | B_ob1280 | D_spin OPPOLL=1"; python ops650_ab.py ops650-h2
} > ops650-final-results.txt 2>&1
{
  echo
  echo "## $(date '+%Y-%m-%d %H:%M')  AUTO-SUMMARY (ops650-finisher.sh): MTP ABBA and host-knob runs; full text in npu_campaign/ops650-final-results.txt"
  echo '```'
  cat ops650-final-results.txt
  echo '```'
} >> OPS650_LOG.md

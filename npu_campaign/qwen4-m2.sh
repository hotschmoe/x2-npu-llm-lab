#!/bin/bash
# master queue 2 (serial): OPPOLL A/B + spin stress + temperature-0.6 matrix. Run after m1.
cd /c/Users/hotschmoe/documents/llm_inference
bash npu_campaign/qwen4-q11.sh      # OPPOLL off/spin/sleep ABBA, 3 rounds, plain decode (pkg s7 with xp+rb)
bash npu_campaign/qwen4-q6.sh 30    # 30 spin runs with the bounded-spin build (pkg s4) -> qwen4-spin-stress.out
bash npu_campaign/qwen4-q9.sh       # MTP vs plain at temp 0.6 (pkg-final2-f1)

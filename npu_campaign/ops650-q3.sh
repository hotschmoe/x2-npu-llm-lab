#!/bin/bash
# ops650 queue 3 (host-side knobs), pkg s11 with fusions OFF: OPBATCH 3072 (1 batch/token) vs 1280 (2 batches/token) vs REPLAY=0 vs OPPOLL=1 (spin).
# Timing envs on. 2 interleaved rounds, plain 128 tokens.
cd /c/Users/hotschmoe/documents/llm_inference
T="LLAMA_DECODE_TIMING=1;GGML_HEXAGON_TIME_BATCH=1"
run() { python npu_campaign/lockrun650.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label "$1" -Spec none -Tokens 128 -DeviceProfile 0 -Pkg 'npu_campaign\pkg-ops650-s11' -ExtraEnv "$T;$2" -Prompt Hello > /dev/null 2>&1; }
declare -A V=( [A_base]="X=1" [B_ob1280]="GGML_HEXAGON_OPBATCH=1280" [C_noreplay]="GGML_HEXAGON_REPLAY=0" [D_spin]="GGML_HEXAGON_OPPOLL=1" )
for r in 1 2; do
  if [ $((r%2)) -eq 1 ]; then ORDER="A_base B_ob1280 C_noreplay D_spin"; else ORDER="D_spin C_noreplay B_ob1280 A_base"; fi
  for k in $ORDER; do run "ops650-h2-$k-r$r" "${V[$k]}"; done
done

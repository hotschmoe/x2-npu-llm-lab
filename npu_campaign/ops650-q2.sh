#!/bin/bash
# ops650 queue 2: baseline profiles + spoof modes (package pkg-ops650-s1). Every run takes the NPU lock via lockrun650.py.
cd /c/Users/hotschmoe/documents/llm_inference
run() { # label spec extraenv tokens
  python npu_campaign/lockrun650.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label "$1" -Spec $2 -Tokens $4 -DeviceProfile 1 -Pkg 'npu_campaign\pkg-ops650-s1' -ExtraEnv "$3" -Prompt Hello
}
run ops650-b-plain none "X=1" 32
run ops650-b-mtp mtp "LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536" 32
for m in 1 2 3 4 5 6 7 8 9 10; do run ops650-sp$m-plain none "GGML_HEXAGON_OPS650_SPOOF=$m" 16; done

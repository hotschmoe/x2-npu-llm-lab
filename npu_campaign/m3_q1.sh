#!/bin/bash
# mtp3 queue 1: fresh baseline ABBA on installed pkg-wos (plain/MTP, block vs spin), Hello 256, 3 rounds
cd /c/Users/hotschmoe/documents/llm_inference
P='llama.cpp-hexagon\pkg-wos'
FCDV='LLAMA_MTP_FUSED_CATCHUP=1,LLAMA_MTP_DRAFT_VOCAB=65536'
powershell -NoProfile -File npu_campaign/mtp-abba.ps1 -Prefix m3b1 -Rounds 3 -Tokens 256 -Prompt Hello \
 -Variants "plain=$P|none|1|;mtp=$P|mtp|1|$FCDV;plainspin=$P|none|1|GGML_HEXAGON_OPPOLL=1;mtpspin=$P|mtp|1|$FCDV,GGML_HEXAGON_OPPOLL=1"

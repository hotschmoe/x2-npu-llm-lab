#!/bin/bash
# master queue 5: GDN state-gather fusion for MTP verify rows (GGML_HEXAGON_GDN_STATE_MAXT=4) on s12: MTP gate (target+draft rows) + profile.
source npu_campaign/qwen4-lib.sh
OUT=npu_campaign/qwen4-m5.out
P='npu_campaign\pkg-qwen4-s13'
ALL="GGML_HEXAGON_MV_XPIPE=1;GGML_HEXAGON_Q4C=3;GGML_HEXAGON_GDN_SNAPDMA=1;GGML_HEXAGON_GDN_STATE_MAXT=4"
MT="LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536"
bash npu_campaign/qwen4-gate.sh s13st "$P" "$ALL" mtp hello | tee -a $OUT
q4run q4-m5-mprof-st "$P" mtp 40 Hello "$ALL;$MT" -DraftMax 1 -DeviceProfile 1 | tee -a $OUT

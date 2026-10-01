#!/bin/bash
source /c/Users/hotschmoe/documents/llm_inference/npu_campaign/m3_lib.sh
PE='npu_campaign\pkg-mtp3-e'
BASH="C:/Program Files/Git/usr/bin/bash.exe"
if [ "$1" = one ]; then wait_ac; echo "## block $2 $(date +%H:%M)"; $LR "$BASH" npu_campaign/m3_blocks8.sh block $2; echo "== done"; exit 0; fi
case $2 in
 prof) $PSF npu_campaign/run-mtp-speed.ps1 -Label m3q-prof-plain -Pkg "$PE" -Spec none -Tokens 24 -DeviceProfile 1 >/dev/null 2>&1
       $PSF npu_campaign/run-mtp-speed.ps1 -Label m3q-prof-mtp -Pkg "$PE" -Spec mtp -DraftMax 1 -Tokens 24 -DeviceProfile 1 -ExtraEnv "$FCDV" -MtpProfile >/dev/null 2>&1
       $PSF npu_campaign/run-mtp-speed.ps1 -Label m3q-prof-mtp2 -Pkg "$PE" -Spec mtp -DraftMax 2 -Tokens 24 -DeviceProfile 1 -ExtraEnv "LLAMA_MTP_DRAFT_VOCAB=65536;LLAMA_MTP_PMIN_NEXT=0.8" >/dev/null 2>&1 ;;
esac

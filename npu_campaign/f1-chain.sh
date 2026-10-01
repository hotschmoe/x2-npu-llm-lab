#!/bin/bash
# final (= rollup + mtp3) gates then timing vs installed; waits for the roll-up mtp-gate to finish; AC-only timing.
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign
until grep -q ALLDONE r1-gates.out 2>/dev/null; do sleep 30; done
for c in 4 16; do powershell -NoProfile -ExecutionPolicy Bypass -File run-prefill-ppl.ps1 -Label f1-q$c -BinDir npu_campaign/pkg-final2-f1/bin -LibDir npu_campaign/pkg-final2-f1/lib -Chunks $c; done
powershell -NoProfile -ExecutionPolicy Bypass -File mtp-gate.ps1 -Prefix f1gate -RefPkg npu_campaign/pkg-merged2-m1 -CandPkg npu_campaign/pkg-final2-f1
echo GATESDONE
pw() { powershell -NoProfile -Command '(Get-CimInstance -Namespace root\WMI -ClassName BatteryStatus).PowerOnline'; }
until [ "$(pw | tr -d '\r')" = "True" ]; do echo "waiting for AC"; sleep 60; done
powershell -NoProfile -ExecutionPolicy Bypass -File ab-multi.ps1 -Tag f1q -Rounds 2 -Specs 'installed|-|-|PF_ROUTER=1','final|npu_campaign/pkg-final2-f1/lib|npu_campaign/pkg-final2-f1/bin|PF_ROUTER=1' 2>&1 | tail -6
powershell -NoProfile -ExecutionPolicy Bypass -File mtp-abba.ps1 -Prefix f1dec -Rounds 2 -Tokens 256 -Variants 'instplain=llama.cpp-hexagon/pkg-wos|none|1|;finplain=npu_campaign/pkg-final2-f1|none|1|;instmtp=llama.cpp-hexagon/pkg-wos|mtp|1|LLAMA_MTP_FUSED_CATCHUP=1,LLAMA_MTP_DRAFT_VOCAB=65536;finmtp=npu_campaign/pkg-final2-f1|mtp|1|LLAMA_MTP_FUSED_CATCHUP=1,LLAMA_MTP_DRAFT_VOCAB=65536;finmtp5=npu_campaign/pkg-final2-f1|mtp|1|LLAMA_MTP_FUSED_CATCHUP=1,LLAMA_MTP_DRAFT_VOCAB=65536|0.5' 2>&1 | tail -10
echo F1DONE

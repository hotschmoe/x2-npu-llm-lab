#!/bin/bash
# wait for AC power and an idle machine, then run the 2048-context pp/tg comparison (installed s14 vs previous build pkg-final2-f1), 2 interleaved rounds
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign
pw() { powershell -NoProfile -Command '(Get-CimInstance -Namespace root\WMI -ClassName BatteryStatus).PowerOnline' | tr -d '\r'; }
until [ "$(pw)" = "True" ]; do sleep 30; done
sleep 30   # let power settle
until [ "$(pw)" = "True" ]; do sleep 30; done
while [ "$(powershell -NoProfile -Command '@(Get-Process llama-* -ErrorAction SilentlyContinue).Count' | tr -d '\r')" != "0" ]; do sleep 10; done
powershell -NoProfile -ExecutionPolicy Bypass -File bench-2048.ps1 -Rounds 2 -Reps 2 > bench-2048-ac.out 2>&1
echo BENCHDONE >> bench-2048-ac.out

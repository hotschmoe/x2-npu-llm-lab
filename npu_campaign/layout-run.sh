#!/bin/bash
# layout-run.sh LABEL PKG TOKENS PROFILE "ENV" [extra run-mtp-speed args]   (lockrun + pre-checks inside the lock)
cd /c/Users/hotschmoe/documents/llm_inference
L=$1; PKG=$2; TOK=$3; PROF=$4; ENVV=$5; shift 5
cat > npu_campaign/layout-inner.$L.sh <<EOT
while true; do
  n=\$(powershell -NoProfile -Command "(Get-Process llama-* -ErrorAction SilentlyContinue | Measure-Object).Count" | tr -d '\r')
  m=\$(powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue" | tr -d '\r')
  [ "\$n" = 0 ] && [ "\$m" -gt 20000 ] && break
  sleep 15
done
powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label $L -Pkg "$PKG" -Tokens $TOK -DeviceProfile $PROF -ExtraEnv "$ENVV" -TimeoutSec 500 $@ > /dev/null 2>&1
EOT
python npu_campaign/lockrun.py -- "C:\Program Files\Git\bin\bash.exe" npu_campaign/layout-inner.$L.sh
rm -f npu_campaign/layout-inner.$L.sh

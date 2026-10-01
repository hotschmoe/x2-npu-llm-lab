#!/bin/bash
# layout-pp.sh LABEL PKGDIR "VAR=VAL or empty" [PP] [UB] [REPS]   prefill bench under lockrun with process/RAM pre-check, retried when killed
cd /c/Users/hotschmoe/documents/llm_inference
L=$1; PK=$2; EV=$3; PP=${4:-1024}; UB=${5:-1024}; RP=${6:-3}
cat > npu_campaign/layout-inner.$L.sh <<EOT
while true; do
  n=\$(powershell -NoProfile -Command "(Get-Process llama-* -ErrorAction SilentlyContinue | Measure-Object).Count" | tr -d '\r')
  m=\$(powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue" | tr -d '\r')
  [ "\$n" = 0 ] && [ "\$m" -gt 20000 ] && break
  sleep 10
done
sleep 8
n=\$(powershell -NoProfile -Command "(Get-Process llama-* -ErrorAction SilentlyContinue | Measure-Object).Count" | tr -d '\r'); [ "\$n" = 0 ] || exit 0
[ -n "$EV" ] && export $EV
powershell -NoProfile -File npu_campaign/run-prefill-bench.ps1 -Label $L -PP $PP -UB $UB -Reps $RP -LibDir "$PK\lib" -BinDir "$PK\bin" > /dev/null 2>&1
EOT
for t in 1 2 3 4 5; do
  rm -f npu_campaign/prefill-$L.log npu_campaign/prefill-$L.log.err
  python npu_campaign/lockrun.py -- "C:\Program Files\Git\bin\bash.exe" npu_campaign/layout-inner.$L.sh
  grep -q 'pp[0-9]* *|' npu_campaign/prefill-$L.log 2>/dev/null && break
  sleep 20
done
rm -f npu_campaign/layout-inner.$L.sh
echo "$L $(grep -o 'pp[0-9]* *| *[0-9.]*' npu_campaign/prefill-$L.log | tr -s ' ' | tr '\n' ' ')"

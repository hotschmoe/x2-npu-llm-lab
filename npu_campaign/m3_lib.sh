# shared helpers for mtp3 queues (source me). run() executes under the NPU lock only if NPU_LOCK_HELD != 1 (blocks hold the lock for several runs).
cd /c/Users/hotschmoe/documents/llm_inference
LR="python npu_campaign/lockrun3.py --"
PSF="powershell -NoProfile -File"
PKG_WOS='llama.cpp-hexagon\pkg-wos'
FCDV='LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536'
D=npu_campaign/mtp-speed
tps() { grep -o '[0-9.]* tokens per second' $D/$1.detail.log 2>/dev/null | tail -1 | cut -d' ' -f1; }
acc() { grep -o '[0-9]* accepted /[ ]*[0-9]* generated' $D/$1.detail.log 2>/dev/null | tail -1 | tr -d ' ' | sed 's/accepted\//\//; s/generated//'; }
# AC / BATT (speed numbers are only valid on AC)
ac() { powershell -NoProfile -Command "if((Get-CimInstance -Namespace root\WMI -ClassName BatteryStatus).PowerOnline){'AC'}else{'BATT'}" | tr -d '\r\n'; }
wait_ac() { while [ "$(ac)" != AC ]; do echo "   (on battery, waiting 60 s)"; sleep 60; done; }
# run LABEL PKG SPEC DMAX TOKENS PROMPT ENV ARGS [PMIN]
run() {
  local l=$1 pkg=$2 spec=$3 dm=$4 tok=$5 pr=$6 env=$7 args=$8 pm=${9:--1}
  if [ "$NPU_LOCK_HELD" = 1 ]; then W=""; else W="$LR"; fi
  local p0; p0=$(ac)
  $W $PSF npu_campaign/run-mtp-speed.ps1 -Label $l -Pkg "$pkg" -Spec $spec -DraftMax $dm -PMin $pm -Tokens $tok -Prompt "$pr" -ExtraEnv "$env" -ExtraArgs "$args" -TimeoutSec ${TMO:-420} > /dev/null 2>&1
  local p1; p1=$(ac); if [ "$p0" != AC ] || [ "$p1" != AC ]; then p1=BATT; fi
  echo "$l tps=$(tps $l) acc=$(acc $l) power=$p1"
}
CODE='Write a Python function that merges overlapping intervals, with a docstring and three unit tests.'
JSON='Give a JSON object for a tool call named get_weather with arguments city Paris and unit celsius, then explain each field.'
SKY='Explain why the sky appears blue in three clear sentences.'
REP='Here is a config: {host: localhost, port: 8080, timeout: 30, retries: 3, workers: 4}. Copy it exactly three times, changing only the port each time (8081, 8082, 8083).'
T06="--temp 0.6 --top-p 0.95 --top-k 20"
PA='npu_campaign\pkg-mtp3-c'
PB='npu_campaign\pkg-mtp3-b'
M6='models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf'
M4='models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32-headq40.gguf'

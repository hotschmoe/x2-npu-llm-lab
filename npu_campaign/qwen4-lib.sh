# qwen4 serial-queue helpers (source me). EVERY NPU run goes through lockrun.py (atomic os.mkdir lock; npu-lock.ps1's New-Item races when several waiters poll in lockstep).
cd /c/Users/hotschmoe/documents/llm_inference
source npu_campaign/m3_lib.sh > /dev/null 2>&1
LK="python npu_campaign/lockrun.py --"
GUARD="python npu_campaign/qwen4-guard.py --"
RUNPS="powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1"
D=npu_campaign/mtp-speed
# q4run LABEL PKG SPEC TOKENS PROMPT ENV [extra ps args...]
q4run() {
  local l=$1 pkg=$2 spec=$3 tok=$4 pr=$5 env=$6; shift 6
  local W="$LK"; [ "$NPU_LOCK_HELD" = 1 ] && W=""
  $W $GUARD $RUNPS -Label "$l" -Pkg "$pkg" -Spec $spec -Tokens $tok -Prompt "$pr" -ExtraEnv "$env" -TimeoutSec ${TMO:-420} "$@" > /dev/null 2>&1
  echo "$l tps=$(tps $l) acc=$(acc $l) power=$(ac)"
}

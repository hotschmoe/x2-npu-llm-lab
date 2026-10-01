# Start the Qwen3.6-35B-A3B Q4_0 llama-server on the Hexagon NPU (ONE consumer, local).
#   serve\start-qwen.cmd -Mode chat        plain decode (~36 tok/s), best for prose / chat
#   serve\start-qwen.cmd -Mode code        MTP d1 + p-min 0.5 (+3..8% on code / JSON / copy-heavy output)
#   serve\start-qwen.cmd -Mode both        = code (llama-server cannot toggle speculation per request; see README)
#   serve\stop-qwen.cmd                    stop server + release NPU lock
#   serve\start-qwen.cmd -Foreground       run in this console (Ctrl-C stops and releases the lock)
#
# The NPU lock (npu_campaign\.npu.lock, shared protocol with the other campaign agents) is held by a small hidden supervisor powershell
# process for the whole life of the server and released when the server exits or stop-qwen is run. The supervisor pid is in the lock's owner.txt.
param(
    [ValidateSet('chat', 'code', 'both')][string]$Mode = 'chat',
    [int]$Port = 8080,
    [string]$BindHost = '127.0.0.1',
    [int]$Ctx = 32768,
    [int]$B = 2048,
    [int]$Ub = 1024,
    [double]$Temp = 0.6, [double]$TopP = 0.95, [int]$TopK = 20, [double]$MinP = 0.0,
    [string]$Reasoning = 'auto',      # llama-server --reasoning: auto | on | off
    [string]$ExtraArgs = '',          # appended verbatim to the llama-server command line
    [string]$Pkg = 'llama.cpp-hexagon\pkg-wos',
    [int]$LockTimeoutSec = 3600,      # how long to queue for the NPU lock
    [int]$StartTimeoutSec = 600,
    [switch]$Foreground,
    [switch]$Supervise                # internal: the detached supervisor re-invokes this script with -Supervise
)
$ErrorActionPreference = 'Stop'
$serveDir = $PSScriptRoot
$root = Split-Path $serveDir -Parent
$runDir = Join-Path $serveDir 'run'; $logDir = Join-Path $serveDir 'logs'
New-Item -ItemType Directory -Force $runDir, $logDir | Out-Null
$pidFile = Join-Path $runDir 'server.pid'; $supFile = Join-Path $runDir 'supervisor.pid'; $modeFile = Join-Path $runDir 'mode.txt'
$statusFile = Join-Path $runDir 'status.txt'

function Test-Health($p) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 3; return ($h.status -eq 'ok') } catch { return $false }
}

# ---------------------------------------------------------------- launcher (default): detach a supervisor, wait for health
if (-not $Supervise) {
    if ((Test-Path $supFile) -and (Get-Process -Id ([int](Get-Content $supFile)) -ErrorAction SilentlyContinue)) {
        throw "A server supervisor is already running (pid $(Get-Content $supFile), mode $(Get-Content $modeFile -ErrorAction SilentlyContinue)). Run serve\stop-qwen.cmd first."
    }
    Remove-Item $statusFile -ErrorAction SilentlyContinue
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Supervise', '-Mode', $Mode, '-Port', $Port, '-BindHost', $BindHost, '-Ctx', $Ctx, '-B', $B, '-Ub', $Ub,
        '-Temp', $Temp, '-TopP', $TopP, '-TopK', $TopK, '-MinP', $MinP, '-Reasoning', $Reasoning, '-Pkg', "`"$Pkg`"", '-LockTimeoutSec', $LockTimeoutSec)
    if ($ExtraArgs) { $argList += @('-ExtraArgs', "`"$ExtraArgs`"") }
    # lockrun.py holds the NPU lock (shared protocol) for the whole life of the inner script = the server
    $lockrun = Join-Path $PSScriptRoot 'lockhold.py'
    $argList = @('-u', "`"$lockrun`"", '--', 'powershell.exe') + $argList
    if ($Foreground) { & python @($argList | ForEach-Object { $_ -replace '^"|"$', '' }); exit $LASTEXITCODE }
    "waiting for NPU lock (held by: $(Get-Content (Join-Path $root 'npu_campaign\.npu.lock\owner.txt') -ErrorAction SilentlyContinue))" | Set-Content $statusFile
    $sup = Start-Process -FilePath 'python' -ArgumentList $argList -WindowStyle Hidden -PassThru
    $sup.Id | Set-Content $supFile
    Write-Host "supervisor pid $($sup.Id); mode=$Mode; waiting for the NPU lock and server (log: serve\logs\server-$Mode.log)"
    $deadline = (Get-Date).AddSeconds($LockTimeoutSec + $StartTimeoutSec)
    $lastMsg = ''
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if (-not (Get-Process -Id $sup.Id -ErrorAction SilentlyContinue)) {
            Write-Host "supervisor exited early: $(Get-Content $statusFile -ErrorAction SilentlyContinue)"; Write-Host "see $logDir"; exit 1
        }
        $st = Get-Content $statusFile -ErrorAction SilentlyContinue
        if ($st -and $st -ne $lastMsg) { Write-Host $st; $lastMsg = $st }
        if (Test-Health $Port) { Write-Host "READY  http://${BindHost}:$Port  (mode $Mode, ctx $Ctx). OpenAI API at /v1, UI at /. Stop: serve\stop-qwen.cmd"; exit 0 }
    }
    Write-Host 'timeout waiting for the server (still queued for the NPU lock or loading); run serve\stop-qwen.cmd to abort'; exit 2
}

# ---------------------------------------------------------------- supervisor / foreground: hold lock, run server
. "$root\npu_campaign\npu-lock.ps1"
# pre-flight (lock is held now): ONE model process on the machine and enough free RAM (ram-guard.py kills a second llama-*.exe)
$other = Get-Process llama-* -ErrorAction SilentlyContinue
$free = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
if ($other -or $free -lt 20000) { "refusing to start: llama-* running=$([bool]$other) availableMB=$([int]$free)" | Set-Content $statusFile; exit 3 }
$pkgDir = (Resolve-Path (Join-Path $root $Pkg)).Path
$exe = Join-Path $pkgDir 'bin\llama-server.exe'
$modelBase = Join-Path $root 'models\gguf\qwen36-35b-a3b-q4_0'
# plain mode uses the plain GGUF; code mode the MTP GGUF (f32 MTP head, ~1 MB bigger) -- same weights otherwise
$gguf = if ($Mode -eq 'chat') { Join-Path $modelBase 'Qwen_Qwen3.6-35B-A3B-Q4_0.gguf' } else { Join-Path $modelBase 'Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf' }
foreach ($f in @($exe, $gguf)) { if (-not (Test-Path $f)) { throw "missing $f" } }

# clean campaign knobs, then set the frozen production decode policy (same as npu_campaign\run-mtp-speed.ps1; OPPOLL is deliberately NOT set: it can hang)
foreach ($v in @('LLAMA_LOGITS_TRACE', 'LLAMA_LOGITS_TRACE_FILE', 'LLAMA_TENSOR_TRACE_FILE', 'LLAMA_TENSOR_TRACE_FILTER', 'LLAMA_RETAIN_TRACE_FILE', 'LLAMA_RETAIN_TRACE_FILTER',
        'LLAMA_RECURRENT_SNAPSHOT_LAYER', 'LLAMA_FUSE_MTP_VERIFY', 'LLAMA_MTP_PAD_ROWS', 'LLAMA_MTP_PMIN_NEXT', 'LLAMA_DECODE_TIMING', 'LLAMA_MTP_PROFILE', 'LLAMA_MTP_KV_ONLY',
        'GGML_HEXAGON_ALLOC_FILL', 'GGML_HEXAGON_NHVX', 'GGML_HEXAGON_OPPOLL', 'GGML_HEXAGON_TAIL_T_MASK', 'GGML_HEXAGON_REPLAY_SLOTS', 'GGML_HEXAGON_TIME_BATCH', 'GGML_HEXAGON_TIME_GET',
        'GGML_HEXAGON_NDEV', 'GGML_HEXAGON_SSM_SCAN', 'LLAMA_MTP_FUSED_CATCHUP', 'LLAMA_MTP_DRAFT_VOCAB')) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
$env:ADSP_LIBRARY_PATH = "$(Join-Path $pkgDir 'lib');"
$env:GGML_HEXAGON_OPBATCH = '3072'; $env:GGML_HEXAGON_REPLAY = '1'; $env:GGML_HEXAGON_OPFUSION = '1'; $env:GGML_HEXAGON_VERBOSE = '0'; $env:GGML_HEXAGON_PROFILE = '0'
$env:GGML_HEXAGON_MOE_TOPK_ALIAS = '0'; $env:GGML_HEXAGON_MM_SELECT = '2'; $env:GGML_HEXAGON_HEAD_PREFETCH = '0'; $env:GGML_HEXAGON_FA_SELECT = '1'; $env:GGML_HEXAGON_GDN_SELECT = '1'
$env:GGML_HEXAGON_ROUTER_HVX = '1'; $env:LLAMA_LOGITS_TRACE = '0'
$spec = '--spec-type none'
if ($Mode -ne 'chat') {
    $env:LLAMA_MTP_FUSED_CATCHUP = '1'; $env:LLAMA_MTP_DRAFT_VOCAB = '65536'
    $spec = '--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.5'
}
$logFile = Join-Path $logDir "server-$Mode.log"
$srvArgs = "-m `"$gguf`" --host $BindHost --port $Port --alias qwen36-35b-a3b --device HTP0 -ngl 99 -fit off --load-mode none " +
        "-c $Ctx -b $B -ub $Ub -t 4 -np 1 --cache-prompt --metrics --jinja --reasoning $Reasoning " +
        "--temp $Temp --top-p $TopP --top-k $TopK --min-p $MinP $spec -cram 2048 $ExtraArgs"

$proc = $null
try {
    "lock acquired; loading model" | Set-Content $statusFile
    $Mode | Set-Content $modeFile
    ("`n=== $(Get-Date -Format o) mode=$Mode`n$exe $srvArgs") | Add-Content $logFile
    $proc = Start-Process -FilePath $exe -ArgumentList $srvArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$logFile.out" -RedirectStandardError "$logFile.err"
    $proc.Id | Set-Content $pidFile
    # a server that dies on its own (or is killed by stop-qwen) ends the supervisor, which releases the lock in the finally block
    $proc.WaitForExit()
    "server exited rc=$($proc.ExitCode)" | Set-Content $statusFile
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    # the process can linger ~15 s after exit while the DSP session / 20 GB are torn down: hold the NPU lock until it is really gone
    if ($proc) { for ($i = 0; $i -lt 120 -and (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Seconds 1 } }
    Remove-Item $pidFile, $modeFile -ErrorAction SilentlyContinue
    Remove-Item $supFile -ErrorAction SilentlyContinue
}

# Host CPU of a loaded-but-idle llama-server on HTP0 vs. during a request (does the burner spin when nothing is queued?).
# Usage: idle-server.ps1 -Label s1 -Pkg npu_campaign\pkg-clover-idle-base [-Env @{...}]
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-idle-base',
    [hashtable]$Env = @{},
    [int]$Port = 18093,
    [double]$IdleSec = 8
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item "env:$k" $Env[$k] }
$out = "$repo\npu_campaign\clover-logs\srv-$Label"
$a = "-m `"$repo\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`" -c 2048 -b 256 -ub 256 -ngl 99 --device HTP0 -t 8 -tb 8 --flash-attn on --port $Port -np 1 --no-warmup"
function Measure-Cpu($p, $sec) {
    $p.Refresh(); $c0 = $p.TotalProcessorTime.TotalSeconds; $k0 = $p.PrivilegedProcessorTime.TotalSeconds; $sw = [Diagnostics.Stopwatch]::StartNew()
    Start-Sleep -Milliseconds ([int]($sec * 1000))
    $p.Refresh(); $dt = $sw.Elapsed.TotalSeconds
    return '{0:F3} cores ({1:F3} kernel) over {2:F1}s' -f (($p.TotalProcessorTime.TotalSeconds - $c0) / $dt), (($p.PrivilegedProcessorTime.TotalSeconds - $k0) / $dt), $dt
}
. "$repo\npu_campaign\npu-lock.ps1"; Enter-NpuLock "idle server $Label"
try {
    $run = Start-Process -FilePath "$pkgDir\bin\llama-server.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    $null = $run.Handle
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not $run.HasExited -and $sw.Elapsed.TotalSeconds -lt 300) {
        if (Select-String -Path "$out.err", "$out.log" -Pattern 'server is listening|HTTP server listening' -Quiet) { break }
        Start-Sleep -Milliseconds 500
    }
    "ready after {0:F0}s" -f $sw.Elapsed.TotalSeconds
    Start-Sleep -Seconds 2
    "idle before any request: " + (Measure-Cpu $run $IdleSec)
    $body = '{"prompt":"Write a long story about a dragon.","n_predict":200,"temperature":0,"top_k":1}'
    $job = Start-Job -ScriptBlock { param($u, $b) Invoke-RestMethod -Uri $u -Method Post -Body $b -ContentType 'application/json' -TimeoutSec 240 } -ArgumentList "http://127.0.0.1:$Port/completion", $body
    Start-Sleep -Seconds 2
    "during request:          " + (Measure-Cpu $run 4)
    $r = Receive-Job $job -Wait -AutoRemoveJob
    "request done: predicted {0} tokens at {1:F2} tok/s" -f $r.timings.predicted_n, $r.timings.predicted_per_second
    Start-Sleep -Seconds 1
    "idle after a request:    " + (Measure-Cpu $run $IdleSec)
} finally {
    if ($run -and -not $run.HasExited) { try { $run.Kill() } catch {} }
    Exit-NpuLock
}

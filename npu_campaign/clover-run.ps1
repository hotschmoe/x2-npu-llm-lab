# Run Clover self-speculative (block-diffusion draft + causal verify) decoding with a pkg-clover-* package.
# Usage: clover-run.ps1 -Label cpu-base -Device none            (CPU reference, no NPU lock)
#        clover-run.ps1 -Label htp-base -Device HTP0 -Env @{GGML_SCHED_DEBUG='2'}
# Logs: npu_campaign\clover-logs\<Label>.log (stdout) / .err (stderr, llama.cpp timing + spec stats).
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-base',
    [string]$Device = 'HTP0',
    [string]$Model = 'models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf',
    [string]$Prompt = 'npu_campaign\clover-prompts\diffusion.txt',
    [int]$NPredict = 256,
    [int]$DraftMax = 15,
    [string]$Accept = '0.9:0.6',   # entropy-threshold acceptance ratio:delta ('' = exact greedy verification)
    [int]$Ctx = 4096,
    [string]$Kv = 'f16',           # KV cache type (README uses bf16; HTP SET_ROWS has no bf16 destination yet)
    [int]$Threads = 8,
    [string]$Exe = 'llama-speculative-simple.exe',
    [string]$Extra = '',
    [hashtable]$Env = @{}
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$logDir = "$PSScriptRoot\clover-logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$ngl = if ($Device -eq 'none') { 0 } else { 99 }
$spec = if ($Exe -eq 'llama-speculative-simple.exe') {
    $acc = if ($Accept) { $r, $d = $Accept.Split(':'); "--spec-accept-ratio $r --spec-accept-delta $d" } else { '' }
    "--spec-type draft-self --spec-draft-n-max $DraftMax $acc"
} else { '' }
$a = "-m `"$repo\$Model`" -f `"$repo\$Prompt`" $spec --temp 0 --top-k 1 --seed 42 -ctk $Kv -ctv $Kv " +
     "-c $Ctx -n $NPredict -b 256 -ub 256 -ngl $ngl --device $Device -t $Threads -tb $Threads --flash-attn on --color off $Extra"
$out = "$logDir\$Label"
"$(Get-Date -Format o) $Exe $a" | Set-Content "$out.cmd"
# CPU-only runs take the lock too: unified LPDDR, a CPU model run steals the NPU's DRAM bandwidth.
$useLock = $true
if ($useLock) { . (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "clover-run.ps1 $Label" }
$sw = [Diagnostics.Stopwatch]::StartNew()
# Host CPU usage is a first-class metric (goal: NPU-only inference leaves the CPU idle for the user).
# Sample the process CPU time every 250 ms; report average busy cores over the generation window.
$samples = [System.Collections.Generic.List[object]]::new()
try {
    $run = Start-Process -FilePath "$pkgDir\bin\$Exe" -ArgumentList $a -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    $null = $run.Handle   # keep the handle so CPU times stay readable after exit
    while (-not $run.HasExited) {
        try { $samples.Add([pscustomobject]@{ t = $sw.Elapsed.TotalSeconds; cpu = $run.TotalProcessorTime.TotalSeconds }) } catch {}
        Start-Sleep -Milliseconds 250
    }
    $run.WaitForExit()
} finally { if ($useLock) { Exit-NpuLock } }
$samples | ForEach-Object { "{0:F2},{1:F3}" -f $_.t, $_.cpu } | Set-Content "$out.cpu.csv"
"EXIT=$($run.ExitCode) WALL=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
$dec = Select-String -Path "$out.err" -Pattern 'decoded +\d+ tokens in +([\d.]+) seconds' | Select-Object -Last 1
$evl = Select-String -Path "$out.err" -Pattern 'eval time = +([\d.]+) ms' | Select-Object -Last 1
$genSec = if ($dec) { [double]$dec.Matches[0].Groups[1].Value } elseif ($evl) { [double]$evl.Matches[0].Groups[1].Value / 1000 } else { 0 }
if ($genSec -gt 1 -and $samples.Count -gt 8) {
    # generation ends just before exit (teardown < ~1 s); window = [last sample - 0.5 - genSec, last sample - 0.5]
    $tEnd = $samples[$samples.Count - 1].t - 0.5; $tBeg = $tEnd - $genSec
    $w = @($samples | Where-Object { $_.t -ge $tBeg -and $_.t -le $tEnd })
    if ($w.Count -ge 2) {
        $cores = ($w[-1].cpu - $w[0].cpu) / ($w[-1].t - $w[0].t)
        "HOST_CPU busy cores during generation = {0:F2} (window {1:F1}s, {2} samples; total process CPU {3:F1}s)" -f $cores, ($w[-1].t - $w[0].t), $w.Count, $samples[-1].cpu
    }
}
Select-String -Path "$out.err" -Pattern 'decoded|accept|n_draft|speed|tok/s|t/s|error|failed|abort' |
    ForEach-Object { $_.Line } | Select-Object -Last 14

# trace agent: llama-perplexity 4x512 KLD of a pkg-clover-* package vs a CPU logit base (or create the base with -MakeBase).
# Usage: trace-kld.ps1 -Label htp-x -Pkg npu_campaign\pkg-clover-base [-Env @{GGML_HEXAGON_OPFILTER='MUL_MAT_ID'}]
#        trace-kld.ps1 -Label qwen-cpu -Device none -MakeBase -Base npu_campaign\clover-logs\qwen36-cpu-kld-base.bin -Model <qwen gguf>
# Logs: npu_campaign\clover-logs\trace-<Label>.{log,err}
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-base',
    [string]$Device = 'HTP0',
    [string]$Model = 'models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf',
    [string]$Base = 'npu_campaign\clover-logs\cpu-kld-base.bin',
    [switch]$MakeBase,
    [int]$Chunks = 4,
    [int]$Threads = 12,
    [string]$Extra = '',
    [hashtable]$Env = @{}
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$ngl = if ($Device -eq 'none') { 0 } else { 99 }
$kld = if ($MakeBase) { "--kl-divergence-base `"$repo\$Base`"" } else { "--kl-divergence-base `"$repo\$Base`" --kl-divergence" }
$a = "-m `"$repo\$Model`" -f `"$repo\npu_campaign\ppl_text.txt`" -c 512 -b 512 -ub 512 --chunks $Chunks -ngl $ngl --device $Device " +
     "-t $Threads -tb $Threads -fit off $kld $Extra"
$out = "$PSScriptRoot\clover-logs\trace-$Label"
"$(Get-Date -Format o) env=$(($Env.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ',') llama-perplexity $a" | Set-Content "$out.cmd"
$useLock = $true   # coordinator rule: CPU model runs also take the lock (shared LPDDR bandwidth / RAM)
if ($useLock) { . (Join-Path $PSScriptRoot 'trace-lock.ps1'); Enter-TraceLock "trace-kld.ps1 $Label" }
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $run = Start-Process -FilePath "$pkgDir\bin\llama-perplexity.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
} finally { if ($useLock) { Exit-TraceLock "trace-kld.ps1 $Label" } }
"EXIT=$($run.ExitCode) WALL=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
Select-String -Path "$out.log", "$out.err" -Pattern 'Final estimate|Mean PPL\(Q\)|Mean    KLD|Same top p|RMS .p|error|abort' |
    ForEach-Object { $_.Line } | Select-Object -Last 8

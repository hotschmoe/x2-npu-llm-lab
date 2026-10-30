# trace agent: run llama-clover-draft-dump (causal prefill + optional non-causal draft forward, logits to a file) with an
# optional tensor trace (LLAMA_TENSOR_TRACE_FILE/_FILTER). Always takes the NPU lock (coordinator rule, CPU runs too).
# Usage: trace-dump.ps1 -Label cpu-c4 -Device none -Tokens npu_campaign\clover-logs\chunk4.tok -AllLogits -DraftN 0 [-Filter '^l_out-']
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-trace-t0',
    [string]$Device = 'HTP0',
    [string]$Model = 'models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf',
    [string]$Tokens = '',
    [string]$Prompt = '',
    [string]$PromptFile = '',
    [string]$Filter = '',
    [int]$DraftN = 16,
    [switch]$AllLogits,
    [string]$StateSave = '',
    [string]$StateLoad = '',
    [string]$Anchor = '',
    [int]$Ub = 512,
    [int]$Ctx = 4096,
    [int]$Threads = 12,
    [string]$Extra = '',
    [hashtable]$Env = @{}
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$dir = "$PSScriptRoot\clover-logs\trace"
New-Item -ItemType Directory -Force $dir | Out-Null
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
$env:CLOVER_DUMP_OUT = "$dir\$Label.logits"
$env:CLOVER_DRAFT_N = "$DraftN"
$env:CLOVER_ALL_LOGITS = if ($AllLogits) { '1' } else { '0' }
if ($Tokens) { $env:CLOVER_TOKENS_FILE = (Resolve-Path (Join-Path $repo $Tokens)).Path }
if ($StateSave) { $env:CLOVER_STATE_SAVE = Join-Path $repo $StateSave }
if ($StateLoad) { $env:CLOVER_STATE_LOAD = (Resolve-Path (Join-Path $repo $StateLoad)).Path }
if ($Anchor) { $env:CLOVER_ANCHOR = $Anchor }
if ($Filter) { $env:LLAMA_TENSOR_TRACE_FILE = "$dir\$Label.trace"; $env:LLAMA_TENSOR_TRACE_FILTER = $Filter }
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$ngl = if ($Device -eq 'none') { 0 } else { 99 }
$p = if ($PromptFile) { "-f `"$repo\$PromptFile`"" } elseif ($Prompt) { "-p `"$Prompt`"" } else { '-p x' }
$a = "-m `"$repo\$Model`" $p -c $Ctx -b $Ub -ub $Ub -ngl $ngl --device $Device -t $Threads -tb $Threads -fit off -ctk f16 -ctv f16 $Extra"
"$(Get-Date -Format o) env=$(($Env.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ',') tokens=$Tokens filter=$Filter draft=$DraftN all=$AllLogits save=$StateSave load=$StateLoad anchor=$Anchor llama-clover-draft-dump $a" | Set-Content "$dir\$Label.cmd"
. (Join-Path $PSScriptRoot 'trace-lock.ps1'); Enter-TraceLock "trace-dump.ps1 $Label"
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $run = Start-Process -FilePath "$pkgDir\bin\llama-clover-draft-dump.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$dir\$Label.log" -RedirectStandardError "$dir\$Label.err"
} finally { Exit-TraceLock "trace-dump.ps1 $Label" }
"EXIT=$($run.ExitCode) WALL=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
Select-String -Path "$dir\$Label.err" -Pattern 'clover-draft-dump:|error|failed|abort' | ForEach-Object { $_.Line } | Select-Object -Last 5

param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Prompt = 'Hello',
    [int]$Tokens = 128,
    [int]$Profile = 0,
    [ValidateSet('HTP0', 'none')][string]$Device = 'HTP0',
    [ValidateSet(0, 1)][int]$Replay = 1,
    [ValidateRange(0, 8190)][int]$Fusion = 1,
    [switch]$RouterHvx,
    [switch]$LogitsTrace,
    [switch]$TopkAlias
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Label must be a simple filename component' }
$campaignRoot = $PSScriptRoot
$repoRoot = Split-Path $campaignRoot -Parent
$env:GGML_HEXAGON_OPBATCH = '3072'
$env:GGML_HEXAGON_REPLAY = "$Replay"
$env:GGML_HEXAGON_OPFUSION = "$Fusion"
$env:LLAMA_LOGITS_TRACE = if ($LogitsTrace) { '1' } else { '0' }
if ($LogitsTrace) {
    $env:LLAMA_LOGITS_TRACE_FILE = "$campaignRoot\$Label.logits.bin"
} else {
    Remove-Item Env:LLAMA_LOGITS_TRACE_FILE -ErrorAction SilentlyContinue
}
Remove-Item Env:LLAMA_TENSOR_TRACE_FILE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_TRACE_FILTER -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_RETAIN_TRACE_FILE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_RETAIN_TRACE_FILTER -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_RECURRENT_SNAPSHOT_LAYER -ErrorAction SilentlyContinue
$env:GGML_HEXAGON_VERBOSE = '0'
$env:GGML_HEXAGON_PROFILE = "$Profile"
$env:GGML_HEXAGON_MOE_TOPK_ALIAS = if ($TopkAlias) { '1' } else { '0' }
$env:GGML_HEXAGON_ROUTER_HVX = if ($RouterHvx) { '1' } else { '0' }
$env:GGML_HEXAGON_MM_SELECT = '2'
$env:GGML_HEXAGON_HEAD_PREFETCH = '0'
$env:GGML_HEXAGON_FA_SELECT = '2'
$env:GGML_HEXAGON_GDN_SELECT = '2'
Remove-Item Env:GGML_HEXAGON_ALLOC_FILL -ErrorAction SilentlyContinue
$env:ADSP_LIBRARY_PATH = "$repoRoot\llama.cpp-hexagon\pkg-wos\lib;"
Remove-Item Env:LLAMA_FUSE_MTP_VERIFY -ErrorAction SilentlyContinue
$exe = "$repoRoot\llama.cpp-hexagon\pkg-wos\bin\llama-completion.exe"
$model = "$repoRoot\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
if ($Prompt.Contains('"')) { throw 'Prompt must not contain double quotes' }
$verbosity = if ($Profile -gt 0) { 5 } else { 4 }
$gpuLayers = if ($Device -eq 'none') { 0 } else { 99 }
$argsText = "-m `"$model`" -n $Tokens -c 256 -b 128 -ub 128 -p `"$Prompt`" --seed 1234 --temp 0 --device $Device -ngl $gpuLayers -fit off --load-mode none -no-cnv --ignore-eos -lv $verbosity"
if ($LogitsTrace) { $argsText += " --log-file `"$campaignRoot\$Label.detail.log`"" }
Write-Output "TOPK_ALIAS=$($env:GGML_HEXAGON_MOE_TOPK_ALIAS) PROFILE=$Profile ARGS=$argsText"
. (Join-Path $PSScriptRoot "npu-lock.ps1"); Enter-NpuLock "run-decode.ps1"
try { $run = Start-Process -FilePath $exe -ArgumentList $argsText -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$campaignRoot\$Label.log" -RedirectStandardError "$campaignRoot\$Label.log.err" } finally { Exit-NpuLock }
Write-Output "RUN_EXIT=$($run.ExitCode)"
if ($run.ExitCode -ne 0) { throw "Decode failed: $($run.ExitCode)" }
Select-String -Path "$campaignRoot\$Label.log.err" -Pattern 'eval time|graphs reused|graph replay hit' | ForEach-Object { $_.Line }

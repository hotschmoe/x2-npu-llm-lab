param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Model = 'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf',
    [string]$PP = '32,128,512',
    [int]$UB = 512,
    [int]$Reps = 2,
    [string]$Extra = '',
    [ValidateSet('HTP0', 'none')][string]$Device = 'HTP0',
    [string]$LibDir = '',
    [string]$BinDir = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$env:GGML_HEXAGON_OPBATCH = '3072'
$env:GGML_HEXAGON_REPLAY = '1'
$env:GGML_HEXAGON_OPFUSION = if ($env:PF_OPFUSION) { $env:PF_OPFUSION } else { '1' }
$env:GGML_HEXAGON_ROUTER_HVX = if ($env:PF_ROUTER) { $env:PF_ROUTER } else { '1' }
if ($env:PF_PROFILE) { $env:GGML_HEXAGON_PROFILE = $env:PF_PROFILE }
$env:GGML_HEXAGON_MM_SELECT = '2'
$env:GGML_HEXAGON_FA_SELECT = '2'
$env:GGML_HEXAGON_GDN_SELECT = '2'
$lib = if ($LibDir) { (Resolve-Path (Join-Path $repoRoot $LibDir)).Path } else { "$repoRoot\llama.cpp-hexagon\pkg-wos\lib" }
$env:ADSP_LIBRARY_PATH = "$lib;"
$binDirResolved = if ($BinDir) { (Resolve-Path (Join-Path $repoRoot $BinDir)).Path } else { "$repoRoot\llama.cpp-hexagon\pkg-wos\bin" }
$exe = "$binDirResolved\llama-bench.exe"
$out = "$PSScriptRoot\prefill-$Label"
$ngl = if ($Device -eq 'none') { 0 } else { 99 }
$a = "-m `"$repoRoot\$Model`" -p $PP -n 0 -ub $UB -b 2048 -r $Reps -ngl $ngl --device $Device -o md $Extra"
"ARGS=$a" | Out-File "$out.log" -Encoding utf8
. (Join-Path $PSScriptRoot "npu-lock-fast.ps1"); Enter-NpuLock "run-prefill-bench.ps1 p2"
try { $run = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.log.err" } finally { Exit-NpuLock }
"EXIT=$($run.ExitCode)"
Get-Content "$out.log"

param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Model = 'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf',
    [string]$PP = '32,128,512',
    [int]$UB = 512,
    [int]$Reps = 2,
    [string]$Extra = '',
    [ValidateSet('HTP0', 'none')][string]$Device = 'HTP0',
    [string]$LibDir = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$env:GGML_HEXAGON_OPBATCH = '3072'
$env:GGML_HEXAGON_REPLAY = '1'
$env:GGML_HEXAGON_OPFUSION = '1'
$env:GGML_HEXAGON_ROUTER_HVX = '1'
if ($env:PF_PROFILE) { $env:GGML_HEXAGON_PROFILE = $env:PF_PROFILE }
$env:GGML_HEXAGON_MM_SELECT = '2'
$env:GGML_HEXAGON_FA_SELECT = '2'
$env:GGML_HEXAGON_GDN_SELECT = '2'
$lib = if ($LibDir) { (Resolve-Path (Join-Path $repoRoot $LibDir)).Path } else { "$repoRoot\llama.cpp-hexagon\pkg-wos\lib" }
$env:ADSP_LIBRARY_PATH = "$lib;"
$exe = "$repoRoot\llama.cpp-hexagon\pkg-wos\bin\llama-bench.exe"
$out = "$PSScriptRoot\prefill-$Label"
$ngl = if ($Device -eq 'none') { 0 } else { 99 }
$a = "-m `"$repoRoot\$Model`" -p $PP -n 0 -ub $UB -b 2048 -r $Reps -ngl $ngl --device $Device -o md $Extra"
"ARGS=$a" | Out-File "$out.log" -Encoding utf8
$run = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.log.err"
"EXIT=$($run.ExitCode)"
Get-Content "$out.log"

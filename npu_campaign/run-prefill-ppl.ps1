# Correctness gate for prefill candidates: perplexity on a fixed text with the candidate skel (or installed one).
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$LibDir = '',
    [int]$Ctx = 512,
    [int]$Chunks = 4,
    [string]$Text = 'npu_campaign\ppl_text.txt'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$env:GGML_HEXAGON_OPBATCH = '3072'; $env:GGML_HEXAGON_REPLAY = '1'; $env:GGML_HEXAGON_OPFUSION = '1'
$env:GGML_HEXAGON_ROUTER_HVX = if ($env:PF_ROUTER) { $env:PF_ROUTER } else { '1' }; $env:GGML_HEXAGON_MM_SELECT = '2'; $env:GGML_HEXAGON_FA_SELECT = '2'; $env:GGML_HEXAGON_GDN_SELECT = '2'
$lib = if ($LibDir) { (Resolve-Path (Join-Path $repoRoot $LibDir)).Path } else { "$repoRoot\llama.cpp-hexagon\pkg-wos\lib" }
$env:ADSP_LIBRARY_PATH = "$lib;"
$exe = "$repoRoot\llama.cpp-hexagon\pkg-wos\bin\llama-perplexity.exe"
$model = "$repoRoot\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
$a = "-m `"$model`" -f `"$repoRoot\$Text`" -c $Ctx -b $Ctx -ub $Ctx --chunks $Chunks -ngl 99 --device HTP0 -fit off"
$out = "$PSScriptRoot\ppl-$Label"
$run = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.log.err"
"EXIT=$($run.ExitCode)"
Select-String -Path "$out.log.err","$out.log" -Pattern 'PPL|Final estimate|perplexity:' | ForEach-Object { $_.Line } | Select-Object -Last 8

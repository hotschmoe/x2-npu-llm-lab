# CPU-backend reference perplexity for the same text/chunks (no NPU).
param([string]$Label = 'cpu', [int]$Ctx = 512, [int]$Chunks = 4, [int]$Threads = 12)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$exe = "$repoRoot\llama.cpp-hexagon\pkg-wos\bin\llama-perplexity.exe"
$model = "$repoRoot\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
$a = "-m `"$model`" -f `"$repoRoot\npu_campaign\ppl_text.txt`" -c $Ctx -b $Ctx -ub $Ctx --chunks $Chunks -ngl 0 --device none -t $Threads -fit off"
$out = "$PSScriptRoot\ppl-$Label"
$run = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.log.err"
"EXIT=$($run.ExitCode)"
Select-String -Path "$out.log.err","$out.log" -Pattern 'Final estimate|seconds per pass' | ForEach-Object { $_.Line } | Select-Object -Last 4

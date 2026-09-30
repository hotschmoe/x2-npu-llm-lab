# upstream sync gates hold B (run via lockrun3.py): MTP gate, Gemma PPL ref vs cand, Lightning PPL ref vs cand, Gemma greedy
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\mamba-std.ps1"
& "$PSScriptRoot\mtp-gate.ps1" -Prefix up1 -RefPkg npu_campaign\pkg-merged2-m1 -CandPkg npu_campaign\pkg-upstream-u1 -Tokens 128 -Keep
$gem = 'models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf'
foreach ($p in 'merged2-m1','upstream-u1') {
  & "$PSScriptRoot\run-prefill-ppl.ps1" -Label "up-gemma-$p" -Model $gem -LibDir "npu_campaign\pkg-$p\lib" -BinDir "npu_campaign\pkg-$p\bin" -Chunks 4 | Select-String 'Final estimate'
}
$L = "$PSScriptRoot\..\models\gguf\nemotron35-lightning-30b-a3b-q4_0\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
foreach ($p in 'merged2-m1','upstream-u1') {
  & "$PSScriptRoot\mamba-run.ps1" -Pkg "pkg-$p" -Exe llama-perplexity -Log "up-light-$p" -TimeoutSec 1500 -Env "$MambaStdEnv" -ArgLine "-m $L -f $PSScriptRoot\ppl_text.txt -c 512 -b 512 -ub 512 --chunks 4 -ngl 99 --device HTP0 -fit off" | Out-Null
  "Lightning PPL $p : " + ((Select-String -Path "$PSScriptRoot\up-light-$p.err" -Pattern 'Final estimate' | ForEach-Object { ($_.Line -split 'Final estimate: ')[1] }) -join '')
}
foreach ($p in 'merged2-m1','upstream-u1') {
  & "$PSScriptRoot\mamba-run.ps1" -Pkg "pkg-$p" -Exe llama-completion -Log "up-gemma-greedy-$p" -TimeoutSec 900 -Env "$MambaStdEnv" -ArgLine "-m $PSScriptRoot\..\$gem -p `"The three primary colors are`" -n 40 --temp 0 -ngl 99 --device HTP0 -no-cnv -fit off -c 1024 -b 512 -ub 512 --no-display-prompt" | Out-Null
  "Gemma greedy $p : " + ((Get-Content "$PSScriptRoot\up-gemma-greedy-$p.log" -Raw) -replace '\s+',' ')
}

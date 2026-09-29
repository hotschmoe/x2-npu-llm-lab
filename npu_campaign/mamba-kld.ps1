# KL-divergence of an HTP0 config vs the frozen CPU logits ($corrname\cpu_nemo.kld, 4x512 of ppl_text.txt).
param([Parameter(Mandatory=$true)][string]$Pkg, [string]$Env = '', [Parameter(Mandatory=$true)][string]$Tag, [int]$Ub = 512)
$corrname = if ($env:MAMBA_CORR) { $env:MAMBA_CORR } else { 'mamba-corr' }
. "$PSScriptRoot\mamba-std.ps1"
$M = if ($env:MAMBA_MODEL) { $env:MAMBA_MODEL } else { 'C:\Users\hotschmoe\documents\llm_inference\models\gguf\nemotron3-nano-30b-a3b-q4_0\nvidia_Nemotron-3-Nano-30B-A3B-Q4_0.gguf' }
$T = "$PSScriptRoot\ppl_text.txt"; $K = "$PSScriptRoot\$corrname\cpu_nemo.kld"
& "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe llama-perplexity -Log "$corrname\kld-$Tag" -TimeoutSec 1800 -Env "$MambaStdEnv;$Env" -ArgLine "-m $M -f $T -c 512 -b 512 -ub $Ub --chunks 4 -ngl 99 --device HTP0 -fit off --kl-divergence-base $K --kl-divergence" | Out-Null
$o = Get-Content "$PSScriptRoot\$corrname\kld-$Tag.log" -Raw
$pick = $o -split "`n" | Where-Object { $_ -match 'Mean PPL\(Q\)|Mean\s+KLD|Same top|RMS Dp|Median\s+KLD|99.0% KLD|Maximum KLD|Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)' }
"== $Tag [$Env]"; $pick | ForEach-Object { $_.Trim() }

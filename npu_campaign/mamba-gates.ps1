# Full gate set for one mamba package: oracles, negative control, PPL, KLD, greedy, Qwen PPL gates.  mamba-gates.ps1 -Pkg pkg-mamba-X -Tag t
param([Parameter(Mandatory=$true)][string]$Pkg, [Parameter(Mandatory=$true)][string]$Tag)
$corrname = if ($env:MAMBA_CORR) { $env:MAMBA_CORR } else { 'mamba-corr' }
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\mamba-std.ps1"
$M = if ($env:MAMBA_MODEL) { $env:MAMBA_MODEL } else { 'C:\Users\hotschmoe\documents\llm_inference\models\gguf\nemotron3-nano-30b-a3b-q4_0\nvidia_Nemotron-3-Nano-30B-A3B-Q4_0.gguf' }
$T = "$PSScriptRoot\ppl_text.txt"
function Summ($f) { (Get-Content $f | ForEach-Object { $_ -replace '\x1b\[[0-9;]*m','' } | Select-String 'tests passed' | ForEach-Object { $_.Line.Trim() }) -join ' ' }
$out = @()
& "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe test-backend-ops -Log "g-$Tag-ssm-exact" -ArgLine 'test -b HTP0 -o SSM_SCAN' -Env 'GGML_HEXAGON_SSM_SCAN=1' | Out-Null
$out += "SSM_SCAN exact (strict 1e-7): " + (Summ "$PSScriptRoot\g-$Tag-ssm-exact.log")
& "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe test-backend-ops -Log "g-$Tag-ssm-hmx" -ArgLine 'test -b HTP0 -o SSM_SCAN' -Env 'GGML_HEXAGON_SSM_SCAN=2;TBO_SSM_HMX_TOL=1' | Out-Null
$out += "SSM_SCAN hmx-ssd (tol 5e-7): " + (Summ "$PSScriptRoot\g-$Tag-ssm-hmx.log")
& "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe test-backend-ops -Log "g-$Tag-ssm-neg" -ArgLine 'test -b HTP0 -o SSM_SCAN' -Env 'GGML_HEXAGON_SSM_SCAN=2;TBO_SSM_HMX_TOL=1;GGML_HEXAGON_SSM_SCAN_SPOIL=1' | Out-Null
$out += "SSM_SCAN NEGATIVE CONTROL (must fail): " + (Summ "$PSScriptRoot\g-$Tag-ssm-neg.log")
& "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe llama-perplexity -Log "g-$Tag-ppl4" -TimeoutSec 1500 -Env "$MambaStdEnv" -ArgLine "-m $M -f $T -c 512 -b 512 -ub 512 --chunks 4 -ngl 99 --device HTP0 -fit off" | Out-Null
$out += "Nemotron PPL 4x512 (CPU 16.0833): " + ((Select-String -Path "$PSScriptRoot\g-$Tag-ppl4.err" -Pattern 'Final estimate' | ForEach-Object { ($_.Line -split 'Final estimate: ')[1] }) -join '')
& "$PSScriptRoot\mamba-kld.ps1" -Pkg $Pkg -Tag "g-$Tag" | Out-Null
$out += "KLD vs CPU: " + ((Select-String -Path "$PSScriptRoot\$corrname\kld-g-$Tag.log" -Pattern 'Mean    KLD|Same top p' | ForEach-Object { ($_.Line -replace '[^\x00-\x7f]','').Trim() }) -join ' | ')
& "$PSScriptRoot\mamba-corr.ps1" -Pkg $Pkg -Tag "g$Tag" | Out-Null
$out += "greedy vs CPU: " + ((& python "$PSScriptRoot\mamba_corr_cmp.py" "$PSScriptRoot\$corrname" "g$Tag") -join "`n")
$repo = Split-Path $PSScriptRoot -Parent
foreach ($ch in 4, 16) {
  & "$repo\npu_campaign\run-prefill-ppl.ps1" -Label "g-$Tag-qwen$ch" -LibDir "npu_campaign\$Pkg\lib" -BinDir "npu_campaign\$Pkg\bin" -Chunks $ch | Out-Null
  $out += "Qwen PPL $ch chunks: " + ((Select-String -Path "$PSScriptRoot\ppl-g-$Tag-qwen$ch.log.err" -Pattern 'Final estimate' | ForEach-Object { ($_.Line -split 'Final estimate: ')[1] }) -join '')
}
$out | Tee-Object -FilePath "$PSScriptRoot\gates-$Tag.txt"

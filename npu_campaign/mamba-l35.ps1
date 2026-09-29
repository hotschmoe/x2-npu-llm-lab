# Nemotron-3.5-Lightning gates on one package: KLD/PPL vs CPU, greedy vs CPU, bench, MTP smoke.
param([Parameter(Mandatory=$true)][string]$Pkg, [string]$Tag = 'l35')
$env:MAMBA_MODEL = 'C:\Users\hotschmoe\documents\llm_inference\models\gguf\nemotron35-lightning-30b-a3b-q4_0\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf'
$env:MAMBA_CORR = 'mamba-corr35'
. "$PSScriptRoot\mamba-std.ps1"
$M = $env:MAMBA_MODEL
& "$PSScriptRoot\mamba-kld.ps1" -Pkg $Pkg -Tag "$Tag" -Env 'GGML_HEXAGON_SSM_SCAN=2' | Out-Null
$out = @("KLD vs CPU (CPU PPL 13.5641): " + ((Select-String -Path "$PSScriptRoot\mamba-corr35\kld-$Tag.log" -Pattern 'Mean PPL\(Q\)  |Mean    KLD|Same top p' | ForEach-Object { ($_.Line -replace '[^\x00-\x7f]','').Trim() }) -join ' | '))
& "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe llama-bench -Log "$Tag-bench" -TimeoutSec 2400 -Env "$MambaStdEnv;GGML_HEXAGON_SSM_SCAN=2" -ArgLine "-m $M -p 128,512,1024 -n 64 -b 2048 -ub 1024 -r 3 -ngl 99 --device HTP0 -o md" | Out-Null
$out += (Get-Content "$PSScriptRoot\$Tag-bench.log" | Where-Object { $_ -match '\|\s+(pp|tg)\d+' } | ForEach-Object { $c = $_ -split '\|'; ($c[8].Trim() + '=' + ($c[9] -replace '[^0-9. ]','' -replace '\s+',' ').Trim()) }) -join '  '
& "$PSScriptRoot\mamba-corr.ps1" -Pkg $Pkg -Tag $Tag -Env 'GGML_HEXAGON_SSM_SCAN=2' | Out-Null
$out += "greedy vs CPU:"; $out += (& python "$PSScriptRoot\mamba_corr_cmp.py" "$PSScriptRoot\mamba-corr35" $Tag)
$out | Tee-Object -FilePath "$PSScriptRoot\gates-$Tag.txt"

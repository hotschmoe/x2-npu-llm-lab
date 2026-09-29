# Interleaved A/B of env variants on the same package: mamba-ab.ps1 -Pkg pkg-mamba-X -Variants @('A=..','B=..') -Rounds 3 [-Tests '-p 512,1024 -n 64']
param([string]$Pkg = '', [Parameter(Mandatory=$true)][string[]]$Variants, [int]$Rounds = 3,
      [string]$Tests = '-p 512,1024 -n 64 -b 2048 -ub 1024 -r 2', [string]$Tag = 'ab')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\mamba-std.ps1"
$M = if ($env:MAMBA_MODEL) { $env:MAMBA_MODEL } else { 'C:\Users\hotschmoe\documents\llm_inference\models\gguf\nemotron3-nano-30b-a3b-q4_0\nvidia_Nemotron-3-Nano-30B-A3B-Q4_0.gguf' }
for ($r = 1; $r -le $Rounds; $r++) {
    for ($i = 0; $i -lt $Variants.Count; $i++) {
        $v = $Variants[$i]
        $pk = $Pkg
        if ($v -match '^(pkg-[A-Za-z0-9_.-]+);?(.*)$') { $pk = $Matches[1]; $v = $Matches[2] }
        $log = "$Tag-v$i-r$r"
        & "$PSScriptRoot\mamba-run.ps1" -Pkg $pk -Exe llama-bench -Log $log -TimeoutSec 1500 -Env "$MambaStdEnv;$v" -ArgLine "-m $M $Tests -ngl 99 --device HTP0 -o md" | Out-Null
        $rows = Get-Content "$PSScriptRoot\$log.log" | Where-Object { $_ -match '\|\s+(pp|tg)\d+' } | ForEach-Object { $c = $_ -split '\|'; ($c[8].Trim() + '=' + ($c[9] -replace '[^0-9. ]','' -replace '\s+',' ').Trim()) }
        "round $r variant $i [$pk $v]: " + ($rows -join '  ')
    }
}

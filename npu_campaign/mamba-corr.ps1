# Greedy-text agreement, CPU vs HTP0, for the prompts in $corrname\. Usage: $corrname.ps1 -Pkg pkg-mamba-X -Env 'A=1;B=2' -Tag t1 [-SkipCpu]
param([Parameter(Mandatory=$true)][string]$Pkg, [string]$Env = '', [string]$Tag = 'corr', [switch]$SkipCpu, [int]$N = 48)
$corrname = if ($env:MAMBA_CORR) { $env:MAMBA_CORR } else { 'mamba-corr' }
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\mamba-std.ps1"
$M = if ($env:MAMBA_MODEL) { $env:MAMBA_MODEL } else { 'C:\Users\hotschmoe\documents\llm_inference\models\gguf\nemotron3-nano-30b-a3b-q4_0\nvidia_Nemotron-3-Nano-30B-A3B-Q4_0.gguf' }
$dir = "$PSScriptRoot\$corrname"
foreach ($f in Get-ChildItem $dir -Filter 'p*.txt' | Sort-Object Name) {
    $b = $f.BaseName
    if (-not $SkipCpu -and -not (Test-Path "$dir\$b.cpu.out")) {
        & "$PSScriptRoot\mamba-run.ps1" -Pkg pkg-mamba-base -Exe llama-completion -Log "$corrname\$b.cpu" -NoLock -TimeoutSec 900 -ArgLine "-m $M -f $($f.FullName) -n $N --temp 0 -ngl 0 --device none -t 8 -no-cnv -fit off -c 4096 -b 512 -ub 512 --no-display-prompt" | Out-Null
        Move-Item "$dir\$b.cpu.log" "$dir\$b.cpu.out" -Force
    }
    & "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe llama-completion -Log "$corrname\$b.$Tag" -TimeoutSec 900 -Env "$MambaStdEnv;$Env" -ArgLine "-m $M -f $($f.FullName) -n $N --temp 0 -ngl 99 --device HTP0 -no-cnv -fit off -c 4096 -b 512 -ub 512 --no-display-prompt" | Out-Null
    Move-Item "$dir\$b.$Tag.log" "$dir\$b.$Tag.out" -Force
}
python "$PSScriptRoot\mamba_corr_cmp.py" $dir $Tag

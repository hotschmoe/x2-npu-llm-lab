# Interleaved A/B of ONE environment variable on the installed (or given) build: alternates VAR=<off> and VAR=<on> per round.
# Usage: ab-env.ps1 -Var GGML_HEXAGON_OPPOLL -Off 0 -On 1 -Rounds 4 -PP 1024 -UB 1024
param(
    [Parameter(Mandatory = $true)][string]$Var,
    [string]$Off = '0',
    [string]$On = '1',
    [int]$Rounds = 4,
    [string]$PP = '1024',
    [int]$UB = 1024,
    [int]$Reps = 3,
    [string]$LibDir = '',
    [string]$BinDir = '',
    [string]$Tag = 'abenv'
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($val in @($Off, $On)) {
        $label = "$Tag-$Var-$val-r$r"
        Set-Item -Path "Env:$Var" -Value $val
        $argv = @('-NoProfile', '-File', "$here\run-prefill-bench.ps1", '-Label', $label, '-PP', $PP, '-UB', $UB, '-Reps', $Reps)
        if ($LibDir) { $argv += @('-LibDir', $LibDir) }
        if ($BinDir) { $argv += @('-BinDir', $BinDir) }
        & powershell @argv | Out-Null
        Remove-Item "Env:$Var" -ErrorAction SilentlyContinue
        foreach ($l in (Select-String -Path "$here\prefill-$label.log" -Pattern 'pp\d+' | ForEach-Object { $_.Line })) {
            if ($l -match '(pp\d+)\s*\|\s*([\d.]+)') { $rows += [pscustomobject]@{ Setting = "$Var=$val"; Round = $r; Test = $Matches[1]; TPS = [double]$Matches[2] } }
        }
    }
}
$rows | Group-Object Setting, Test | ForEach-Object {
    $v = $_.Group | Measure-Object TPS -Average -Minimum -Maximum
    [pscustomobject]@{ Setting = $_.Group[0].Setting; Test = $_.Group[0].Test; Mean = [math]::Round($v.Average, 1); Min = [math]::Round($v.Minimum, 1); Max = [math]::Round($v.Maximum, 1); N = $v.Count }
} | Sort-Object Test, Setting | Format-Table -AutoSize

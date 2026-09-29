# Interleaved sweep of ONE environment variable over several values (rounds x values), for a given skel dir / host bin dir.
# Usage: ab-vals.ps1 -Var GGML_HEXAGON_NHVX -Values 8,6,7 -Rounds 3 -LibDir npu_campaign\prefill2-cand-base
param(
    [Parameter(Mandatory = $true)][string]$Var,
    [Parameter(Mandatory = $true)][string[]]$Values,
    [int]$Rounds = 3, [string]$PP = '1024', [int]$UB = 1024, [int]$Reps = 3,
    [string]$LibDir = '', [string]$BinDir = '', [string]$Tag = 'abvals', [string]$Model = ''
)
$ErrorActionPreference = 'Stop'
$Values = @($Values | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
$here = $PSScriptRoot
$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($val in $Values) {
        $label = "$Tag-$val-r$r"
        Set-Item -Path "Env:$Var" -Value $val
        $argv = @('-NoProfile', '-File', "$here\run-prefill-bench.ps1", '-Label', $label, '-PP', $PP, '-UB', $UB, '-Reps', $Reps)
        if ($LibDir) { $argv += @('-LibDir', $LibDir) }
        if ($BinDir) { $argv += @('-BinDir', $BinDir) }
        if ($Model) { $argv += @('-Model', $Model) }
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

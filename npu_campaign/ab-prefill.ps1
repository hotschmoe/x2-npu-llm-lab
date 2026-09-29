# Interleaved A/B for prefill candidates: rounds x variants, each a fresh llama-bench process (-r reps).
# Usage: ab-prefill.ps1 -Variants base,pipe,pipe2,pipe3 -Rounds 3 -PP 512 -UB 512
param(
    [Parameter(Mandatory = $true)][string[]]$Variants,
    [int]$Rounds = 3,
    [string]$PP = '512',
    [int]$UB = 512,
    [int]$Reps = 3,
    [string]$Tag = 'ab'
)
$ErrorActionPreference = 'Stop'
$Variants = @($Variants | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$here = $PSScriptRoot
$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($v in $Variants) {
        # variant syntax: name  or  name:pkgdir  (pkgdir = folder under llama.cpp-hexagon holding bin/, e.g. pkg-prefill)
        $vname = ($v -split ':')[0]
        $pkg = if ($v -like '*:*') { ($v -split ':')[1] } else { '' }
        $label = "$Tag-$vname-r$r"
        $dir = if ($vname -eq 'installed') { '' } else { "npu_campaign\prefill-cand-$vname" }
        $bin = if ($pkg) { "llama.cpp-hexagon\$pkg\bin" } else { '' }
        $argv = @('-NoProfile', '-File', "$here\run-prefill-bench.ps1", '-Label', $label, '-PP', $PP, '-UB', $UB, '-Reps', $Reps)
        if ($dir) { $argv += @('-LibDir', $dir) }
        if ($bin) { $argv += @('-BinDir', $bin) }
        & powershell @argv | Out-Null
        $line = Select-String -Path "$here\prefill-$label.log" -Pattern 'pp\d+' | ForEach-Object { $_.Line }
        foreach ($l in $line) {
            if ($l -match '(pp\d+)\s*\|\s*([\d.]+)') { $rows += [pscustomobject]@{ Variant = $v; Round = $r; Test = $Matches[1]; TPS = [double]$Matches[2] } }
        }
    }
}
$rows | Group-Object Variant, Test | ForEach-Object {
    $v = $_.Group | Measure-Object TPS -Average -Minimum -Maximum
    [pscustomobject]@{ Variant = $_.Group[0].Variant; Test = $_.Group[0].Test; Mean = [math]::Round($v.Average, 1); Min = [math]::Round($v.Minimum, 1); Max = [math]::Round($v.Maximum, 1); N = $v.Count }
} | Sort-Object Test, Variant | Format-Table -AutoSize

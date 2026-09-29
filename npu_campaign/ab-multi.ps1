# Interleaved A/B over arbitrary configs. Each spec: label|libdir(or -)|bindir(or -)|VAR=VAL;VAR=VAL  (env applied only for that run)
# Example: ab-multi.ps1 -Specs 'base|npu_campaign\prefill2-cand-base|-|PF_ROUTER=1','rr|npu_campaign\prefill2-cand-fix1|llama.cpp-hexagon-prefill2\pkg-prefill2\bin|PF_ROUTER=0' -Rounds 3
param(
    [Parameter(Mandatory = $true)][string[]]$Specs,
    [int]$Rounds = 3, [string]$PP = '1024', [int]$UB = 1024, [int]$Reps = 3, [string]$Tag = 'abm', [string]$Model = ''
)
$ErrorActionPreference = 'Stop'
$Specs = @($Specs | ForEach-Object { $_ })
$here = $PSScriptRoot
$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($sp in $Specs) {
        $f = $sp -split '\|'
        $name = $f[0]; $lib = $f[1]; $bin = $f[2]; $envs = if ($f.Count -gt 3) { $f[3] } else { '' }
        $label = "$Tag-$name-r$r"
        $saved = @{}
        foreach ($kv in ($envs -split ';' | Where-Object { $_ -match '=' })) {
            $k, $v = $kv -split '=', 2
            $saved[$k] = [Environment]::GetEnvironmentVariable($k)
            [Environment]::SetEnvironmentVariable($k, $v)
        }
        $argv = @('-NoProfile', '-File', "$here\run-prefill-bench.ps1", '-Label', $label, '-PP', $PP, '-UB', $UB, '-Reps', $Reps)
        if ($lib -and $lib -ne '-') { $argv += @('-LibDir', $lib) }
        if ($bin -and $bin -ne '-') { $argv += @('-BinDir', $bin) }
        if ($Model) { $argv += @('-Model', $Model) }
        & powershell @argv | Out-Null
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
        foreach ($l in (Select-String -Path "$here\prefill-$label.log" -Pattern 'pp\d+' | ForEach-Object { $_.Line })) {
            if ($l -match '(pp\d+)\s*\|\s*([\d.]+)') { $rows += [pscustomobject]@{ Variant = $name; Round = $r; Test = $Matches[1]; TPS = [double]$Matches[2] } }
        }
    }
}
$rows | Group-Object Variant, Test | ForEach-Object {
    $v = $_.Group | Measure-Object TPS -Average -Minimum -Maximum
    [pscustomobject]@{ Variant = $_.Group[0].Variant; Test = $_.Group[0].Test; Mean = [math]::Round($v.Average, 1); Min = [math]::Round($v.Minimum, 1); Max = [math]::Round($v.Maximum, 1); N = $v.Count }
} | Sort-Object Test, Variant | Format-Table -AutoSize

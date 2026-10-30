# One lock hold: interleaved ABAB speed A/B (clover-run, exact greedy verify) base vs candidate package.
param([string]$PkgA = 'npu_campaign\pkg-clover-base', [string]$PkgB = 'npu_campaign\pkg-clover-bf16-a', [string]$Tag = 'ab', [int]$Rounds = 2, [string]$EnvB = '', [string]$EnvA = '')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-bundleC $Tag"
$env:NPU_LOCK_HELD = '1'
try {
    for ($r = 1; $r -le $Rounds; $r++) {
        foreach ($side in 'A', 'B') {
            $pkg = if ($side -eq 'A') { $PkgA } else { $PkgB }
            $e = @{}
            if ($side -eq 'B' -and $EnvB) { foreach ($kv in $EnvB.Split(',')) { $k, $v = $kv.Split('='); $e[$k] = $v } }
            if ($side -eq 'A' -and $EnvA) { foreach ($kv in $EnvA.Split(',')) { $k, $v = $kv.Split('='); $e[$k] = $v } }
            "--- $Tag $side r$r $pkg"
            & "$PSScriptRoot\clover-run.ps1" -Pkg $pkg -Label "bf16-$Tag-$side$r" -Device HTP0 -Accept '' -Extra '-v' -Env $e
            # clover-run leaves its env in this process: clear what we set
            foreach ($k in $e.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
        }
    }
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

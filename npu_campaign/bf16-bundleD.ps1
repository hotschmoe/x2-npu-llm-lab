# One lock hold: a list of short clover-run configs (determinism / toggle diagnostics). Spec: "label|pkgdir|K=V,K=V|npredict"
param([string]$Runs, [string]$Tag = 'd')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-bundleD $Tag"
$env:NPU_LOCK_HELD = '1'
try {
    foreach ($spec in $Runs.Split(";")) {
        $label, $pkg, $envs, $np = $spec.Split('|')
        $e = @{}
        if ($envs) { foreach ($kv in $envs.Split(',')) { $k, $v = $kv.Split('='); $e[$k] = $v } }
        "--- $label $pkg $envs"
        & "$PSScriptRoot\clover-run.ps1" -Pkg $pkg -Label "bf16-$label" -Device HTP0 -Accept '' -NPredict ([int]$np) -Extra '-v' -Env $e
        foreach ($k in $e.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
    }
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

# bf16 agent: several traced HTP dumps (different sync-point filters) in ONE lock hold (hazard bisection).
param([string]$Pkg = 'npu_campaign\pkg-clover-bf16-g', [string]$Tag = 'v', [string]$Filters, [string]$EnvSpec = '')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-tracevar $Tag"
$env:NPU_LOCK_HELD = '1'
try {
    $e = @{}
    if ($EnvSpec) { foreach ($kv in $EnvSpec.Split(',')) { $k, $v = $kv.Split('='); $e[$k] = $v } }
    $i = 0
    foreach ($f in $Filters.Split(';')) {
        $i++
        & "$PSScriptRoot\trace-dump.ps1" -Label "bf16-$Tag$i" -Pkg $Pkg -PromptFile 'npu_campaign\clover-prompts\diffusion.txt' -DraftN 0 -Filter $f -Env $e
    }
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

param([string]$Pkg = 'pkg-clover-bf16-h', [int]$Reps = 3, [string]$Tag = 'h')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-stack $Tag"
$env:NPU_LOCK_HELD = '1'
try {
    for ($i = 1; $i -le $Reps; $i++) {
        & "$PSScriptRoot\bf16-tbo.ps1" -Pkg $Pkg -Mode test -Op CPY_BF16_STACK -Params 'n_layers' -Label "bf16-stack-$Tag-$i"
        Get-Content "$PSScriptRoot\clover-logs\bf16-stack-$Tag-$i.log" | Select-String 'CPY_BF16_STACK|ERR|passed' | ForEach-Object { ($_.Line -replace '\x1b\[[0-9;]*m', '') }
    }
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

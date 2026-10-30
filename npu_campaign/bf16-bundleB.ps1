# One lock hold: KLD gate of a package vs the CPU reference logits.
param([string]$Pkg = 'pkg-clover-bf16-a', [string]$Tag = 'a', [string]$Fuse = '')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-bundleB kld $Tag"
$env:NPU_LOCK_HELD = '1'
try {
    $e = @{}
    if ($Fuse) { $e['GGML_HEXAGON_OPFUSION'] = $Fuse }
    & "$PSScriptRoot\clover-kld.ps1" -Label "bf16-$Tag" -Pkg "npu_campaign\$Pkg" -Env $e
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

# One lock hold: fusion-engagement check (verbose), oracle suite (fused / unfused / spoil), placement audit run.
param([string]$Pkg = 'pkg-clover-bf16-a', [string]$Tag = 'a3')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-bundleA $Tag"
$env:NPU_LOCK_HELD = '1'
try {
    & "$PSScriptRoot\bf16-tbo.ps1" -Pkg $Pkg -Mode test -Op CPY_BF16_ROUND -Params 'ne=\[2048,256' -Label "bf16-fv-on-$Tag" -Env @{GGML_HEXAGON_VERBOSE='1'}
    & "$PSScriptRoot\bf16-tbo.ps1" -Pkg $Pkg -Mode test -Op CPY_BF16_ROUND -Params 'ne=\[2048,256' -Label "bf16-fv-off-$Tag" -Env @{GGML_HEXAGON_VERBOSE='1'; GGML_HEXAGON_OPFUSION='8190'}
    & "$PSScriptRoot\bf16-suite.ps1" -Pkg $Pkg -Tag $Tag
    & "$PSScriptRoot\clover-run.ps1" -Pkg "npu_campaign\$Pkg" -Label "bf16-sched-$Tag" -Device HTP0 -Accept '' -NPredict 8 -Extra '-v' -Env @{GGML_SCHED_DEBUG='2'}
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

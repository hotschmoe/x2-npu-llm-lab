# One lock hold: GDN-front-end synthetic test (default x3, OPBATCH=1 control) + two clean 16-token determinism runs.
param([string]$Pkg = 'pkg-clover-bf16-i')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-hold2"
$env:NPU_LOCK_HELD = '1'
try {
    $stray = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'llama-*' -or $_.ProcessName -eq 'test-backend-ops' }
    "stray HTP processes at start: $(@($stray).Count) $((@($stray) | ForEach-Object { $_.ProcessName + ':' + $_.Id }) -join ',')"
    for ($i = 1; $i -le 3; $i++) {
        & "$PSScriptRoot\bf16-tbo.ps1" -Pkg $Pkg -Mode test -Op CPY_BF16_GDN -Params 'T=' -Label "bf16-gdn-$i" | Out-Null
        Get-Content "$PSScriptRoot\clover-logs\bf16-gdn-$i.log" | Select-String 'CPY_BF16_GDN|ERR|passed' | ForEach-Object { ($_.Line -replace '\x1b\[[0-9;]*m', '') }
    }
    & "$PSScriptRoot\bf16-tbo.ps1" -Pkg $Pkg -Mode test -Op CPY_BF16_GDN -Params 'T=' -Label "bf16-gdn-ob1" -Env @{GGML_HEXAGON_OPBATCH='1'} | Out-Null
    "-- OPBATCH=1"
    Get-Content "$PSScriptRoot\clover-logs\bf16-gdn-ob1.log" | Select-String 'CPY_BF16_GDN|ERR|passed' | ForEach-Object { ($_.Line -replace '\x1b\[[0-9;]*m', '') }
    foreach ($r in 1, 2) {
        "-- det run $r"
        & "$PSScriptRoot\clover-run.ps1" -Pkg "npu_campaign\$Pkg" -Label "bf16-clean-det$r" -Device HTP0 -Accept '' -NPredict 16 -Extra '-v'
    }
} finally { $env:NPU_LOCK_HELD = '0'; Exit-NpuLock }

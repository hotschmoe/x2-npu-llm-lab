# Run test-backend-ops from a pkg-clover-topk-* package under the NPU lock. Output -> clover-logs\<Label>.log
# Usage: topk-tbo.ps1 -Label tbo-test -Pkg npu_campaign\pkg-clover-topk-a -TboArgs 'test -b HTP0 -o TOP_K' -Env @{GGML_HEXAGON_TOPK_SPOIL='1'}
param([Parameter(Mandatory=$true)][string]$Label, [string]$Pkg = 'npu_campaign\pkg-clover-topk-a',
      [string]$TboArgs = 'test -b HTP0 -o TOP_K', [hashtable]$Env = @{})
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$logDir = "$PSScriptRoot\clover-logs"; New-Item -ItemType Directory -Force $logDir | Out-Null
. (Join-Path $PSScriptRoot 'topk-lockutil.ps1'); Enter-TopkLock "topk-tbo.ps1 $Label"
try {
    $p = Start-Process -FilePath "$pkgDir\bin\test-backend-ops.exe" -ArgumentList $TboArgs -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$logDir\$Label.log" -RedirectStandardError "$logDir\$Label.err"
} finally { Exit-NpuLock }
"EXIT=$($p.ExitCode)"
Select-String -Path "$logDir\$Label.log" -Pattern 'Backend .*:|tests passed|FAIL|OK$' | Select-Object -Last 8 | ForEach-Object { $_.Line }

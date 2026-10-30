# One NPU-lock hold: the Clover CPY oracle in three configs (fused default, unfused via OPFUSION mask 8190, spoil negative control) + perf.
param([string]$Pkg, [string]$Tag = 'x', [switch]$NoPerf)
$ErrorActionPreference = 'Stop'
$pkgDir = (Resolve-Path (Join-Path $PSScriptRoot $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-suite $Tag"
function Run([string]$Label, [string[]]$A, [hashtable]$E) {
    $E['GGML_HEXAGON_BF16_CPY'] = '1'   # bf16 CPY on HTP is opt-in
    $saved = @{}
    foreach ($k in $E.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $E[$k]) }
    $out = "$PSScriptRoot\clover-logs\$Label"
    $p = Start-Process -FilePath "$pkgDir\bin\test-backend-ops.exe" -ArgumentList $A -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    $l = Get-Content "$out.log"
    $ok = @($l | Select-String 'OK' | Where-Object { $_.Line -match 'mode=' }).Count
    $fail = @($l | Select-String 'FAIL' | Where-Object { $_.Line -match 'mode=' }).Count
    $ns = @($l | Select-String 'not supported' | Where-Object { $_.Line -match 'mode=' }).Count
    "$Label EXIT=$($p.ExitCode) cases OK=$ok FAIL=$fail NOTSUP=$ns"
}
try {
    Run "bf16-oracle-$Tag-fused"   @('test', '-b', 'HTP0', '-o', 'CPY,CPY_BF16_ROUND', '-p', 'mode=') @{}
    Run "bf16-oracle-$Tag-unfused" @('test', '-b', 'HTP0', '-o', 'CPY,CPY_BF16_ROUND', '-p', 'mode=') @{ GGML_HEXAGON_OPFUSION = '8190' }
    Run "bf16-oracle-$Tag-spoil"   @('test', '-b', 'HTP0', '-o', 'CPY,CPY_BF16_ROUND', '-p', 'mode=') @{ GGML_HEXAGON_BF16_SPOIL = '1' }
    if (-not $NoPerf) {
        Run "bf16-perf-$Tag-fused"   @('perf', '-b', 'HTP0', '-o', 'CPY,CPY_BF16_ROUND', '-p', 'mode=') @{}
        Run "bf16-perf-$Tag-unfused" @('perf', '-b', 'HTP0', '-o', 'CPY,CPY_BF16_ROUND', '-p', 'mode=') @{ GGML_HEXAGON_OPFUSION = '8190' }
    }
} finally { Exit-NpuLock }

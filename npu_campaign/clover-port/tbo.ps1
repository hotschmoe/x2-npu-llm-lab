# Run test-backend-ops (test or perf) from a pkg-clover-port-* package inside the NPU lock.
# Usage: tbo.ps1 -Label x -Pkg npu_campaign\pkg-clover-port-pad -Mode test|perf -Op MUL_MAT_ID [-Filter 'type_a=q4_0']
param([Parameter(Mandatory)][string]$Label,[Parameter(Mandatory)][string]$Pkg,[string]$Mode='test',[string]$Op='MUL_MAT_ID',[string]$Filter='',[hashtable]$Env=@{})
$ErrorActionPreference='Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$out = "$repo\npu_campaign\clover-logs\tbo-$Label"
$a = "$Mode -b HTP0 -o $Op"
if ($Filter) { $a += " -p `"$Filter`"" }
. "$repo\npu_campaign\npu-lock.ps1"; Enter-NpuLock "tbo.ps1 $Label"
try {
    $run = Start-Process -FilePath "$pkgDir\bin\test-backend-ops.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    if (-not $run.WaitForExit(540000)) { Stop-Process -Id $run.Id -Force; 'TIMEOUT 540 s, killed' }
} finally { if ($run -and -not $run.HasExited) { Stop-Process -Id $run.Id -Force -ErrorAction SilentlyContinue }; Exit-NpuLock }
"EXIT=$($run.ExitCode)"
Select-String -Path "$out.log" -Pattern 'passed|FAIL|Backend .* OK|us/run' | Select-Object -Last 40 | ForEach-Object { $_.Line }

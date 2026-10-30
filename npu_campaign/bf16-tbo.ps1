# Run test-backend-ops from a pkg-clover-bf16-* package under the NPU lock (oracle / perf for the Clover CPY cases).
# Usage: bf16-tbo.ps1 -Pkg pkg-clover-bf16-a -Mode test -Op CPY [-Params 'mode=pair'] [-Env @{GGML_HEXAGON_BF16_SPOIL='1'}] -Label x
param([string]$Pkg, [string]$Mode = 'test', [string]$Op = 'CPY', [string]$Params = '', [hashtable]$Env = @{}, [string]$Label = 'tbo', [string]$Backend = 'HTP0')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $PSScriptRoot $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
$saved = @{}
foreach ($k in $Env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); Set-Item -Path "env:$k" -Value $Env[$k] }
$out = "$PSScriptRoot\clover-logs\$Label"
$a = @($Mode, '-b', $Backend, '-o', $Op)
if ($Params) { $a += @('-p', $Params) }
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "bf16-tbo $Label"
try {
    $p = Start-Process -FilePath "$pkgDir\bin\test-backend-ops.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
} finally { Exit-NpuLock; foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
"EXIT=$($p.ExitCode)"

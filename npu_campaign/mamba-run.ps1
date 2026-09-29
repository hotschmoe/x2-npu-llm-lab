# Run a llama.cpp tool from a mamba package under the NPU lock (unless -NoLock / CPU-only).
# Usage: mamba-run.ps1 -Pkg pkg-mamba-X -Exe test-backend-ops -Log name -Args '...' [-Env 'A=1;B=2'] [-NoLock]
param([Parameter(Mandatory=$true)][string]$Pkg, [Parameter(Mandatory=$true)][string]$Exe, [Parameter(Mandatory=$true)][string]$Log,
      [string]$ArgLine = '', [string]$Env = '', [switch]$NoLock, [int]$TimeoutSec = 3600)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not [System.IO.Path]::IsPathRooted($Pkg)) { $Pkg = Join-Path $PSScriptRoot $Pkg }
. "$PSScriptRoot\npu-lock.ps1"
$saved = @{}
foreach ($kv in ($Env -split ';' | Where-Object { $_ })) {
    $k, $v = $kv -split '=', 2
    $saved[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $v)
}
[Environment]::SetEnvironmentVariable('ADSP_LIBRARY_PATH', "$Pkg\lib;")
if (-not $NoLock) {
    # same lock dir/owner protocol as npu-lock.ps1 but polls every 50 ms so that a chain of back-to-back holders cannot starve us
    $dl = (Get-Date).AddSeconds(14400)
    while ($true) {
        try { New-Item -ItemType Directory -Path $script:NpuLockDir -ErrorAction Stop | Out-Null; "mamba:$Log pid=$PID started=$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $script:NpuLockDir 'owner.txt'); break }
        catch {
            $f = Join-Path $script:NpuLockDir 'owner.txt'
            if (Test-Path -LiteralPath $f) {
                $ownerPid = ((Get-Content -LiteralPath $f -ErrorAction SilentlyContinue) -replace '.*pid=(\d+).*', '$1')
                $alive = $ownerPid -match '^\d+$' -and (Get-Process -Id ([int]$ownerPid) -ErrorAction SilentlyContinue)
                $fi = Get-Item -LiteralPath $f -ErrorAction SilentlyContinue
                if ($fi -and -not $alive -and $fi.LastWriteTime -lt (Get-Date).AddMinutes(-2)) { Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue; continue }
            }
            if ((Get-Date) -gt $dl) { throw 'NPU lock timeout' }
            # spin until the lock dir disappears (holders hand over within ~10 ms; a sleeping poller never wins)
            $sp = [Diagnostics.Stopwatch]::StartNew()
            while ([System.IO.Directory]::Exists($script:NpuLockDir) -and $sp.ElapsedMilliseconds -lt 2000) { }
        }
    }
}
try {
    $out = Join-Path $PSScriptRoot $Log
    $p = Start-Process -FilePath "$Pkg\bin\$Exe.exe" -ArgumentList $ArgLine -PassThru -WindowStyle Hidden -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { $p.Kill(); "TIMEOUT" }
    "EXIT=$($p.ExitCode) log=$out.log"
} finally {
    if (-not $NoLock) { Exit-NpuLock }
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
}

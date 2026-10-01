# Cross-agent NPU mutex: only one inference/benchmark process on the NPU at a time (timings are otherwise garbage).
# Usage (dot-source): . "$PSScriptRoot\npu-lock.ps1"; Enter-NpuLock 'owner-label'; try { ... } finally { Exit-NpuLock }
$script:NpuLockDir = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
function Enter-NpuLock {
    param([string]$Owner = 'unknown', [int]$TimeoutSec = 14400)
    if ($env:NPU_LOCK_HELD -eq '1') { return }   # caller (lockrun.py) already holds the lock
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ($true) {
        try {
            New-Item -ItemType Directory -Path $script:NpuLockDir -ErrorAction Stop | Out-Null
            "$Owner pid=$PID started=$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $script:NpuLockDir 'owner.txt')
            return
        } catch {
            $f = Join-Path $script:NpuLockDir 'owner.txt'
            if (Test-Path -LiteralPath $f) {
                $t = (Get-Item -LiteralPath $f).LastWriteTime
                $ownerPid = ((Get-Content -LiteralPath $f) -replace '.*pid=(\d+).*', '$1')
                $alive = $ownerPid -match '^\d+$' -and (Get-Process -Id ([int]$ownerPid) -ErrorAction SilentlyContinue)
                if (-not $alive -and $t -lt (Get-Date).AddMinutes(-2)) { Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue; continue }
            }
            if ((Get-Date) -gt $deadline) { throw "NPU lock timeout (held by: $(Get-Content -LiteralPath $f -ErrorAction SilentlyContinue))" }
            Start-Sleep -Milliseconds 200
        }
    }
}
function Exit-NpuLock { if ($env:NPU_LOCK_HELD -eq '1') { return }; Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue }

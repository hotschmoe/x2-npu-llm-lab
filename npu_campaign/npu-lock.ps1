# Cross-agent NPU mutex: only one inference/benchmark process on the NPU at a time (timings are otherwise garbage).
# Usage (dot-source): . "$PSScriptRoot\npu-lock.ps1"; Enter-NpuLock 'owner-label'; try { ... } finally { Exit-NpuLock }
# Protocol (shared with lockrun.py): the lock is the directory .npu.lock, created atomically; owner.txt inside names the holder ("label pid=N started=...").
# Atomicity: the directory is created with the Win32 CreateDirectoryW call, which FAILS if the directory exists (New-Item/[Directory]::CreateDirectory are
# check-then-create / idempotent and let two waiters both "win"). A stale lock (owner process dead, owner.txt older than 2 min) is taken over by an atomic
# rename of the whole directory, so only one waiter can claim it, and a freshly created lock is never deleted by a waiter that read old data.
$script:NpuLockDir = if ($env:NPU_LOCK_DIR_TEST) { $env:NPU_LOCK_DIR_TEST } else { 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock' }   # NPU_LOCK_DIR_TEST: tests only
if (-not ('NpuLockNative' -as [type])) {
    Add-Type -Name NpuLockNative -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)] public static extern bool CreateDirectoryW(string path, System.IntPtr sa);'
}
function Test-NpuLockCreate {
    # true only for the single caller that actually created the directory
    return [bool][NpuLockNative]::CreateDirectoryW($script:NpuLockDir, [IntPtr]::Zero)
}
function Enter-NpuLock {
    param([string]$Owner = 'unknown', [int]$TimeoutSec = 14400)
    if ($env:NPU_LOCK_HELD -eq '1') { return }   # caller (lockrun.py) already holds the lock
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $f = Join-Path $script:NpuLockDir 'owner.txt'
    while ($true) {
        if (Test-NpuLockCreate) {
            "$Owner pid=$PID started=$(Get-Date -Format o)" | Set-Content -LiteralPath $f
            return
        }
        if (Test-Path -LiteralPath $f) {
            $t = (Get-Item -LiteralPath $f -ErrorAction SilentlyContinue).LastWriteTime
            $line = (Get-Content -LiteralPath $f -ErrorAction SilentlyContinue | Select-Object -First 1)
            $ownerPid = ($line -replace '.*pid=(\d+).*', '$1')
            $alive = $ownerPid -match '^\d+$' -and (Get-Process -Id ([int]$ownerPid) -ErrorAction SilentlyContinue)
            if (-not $alive -and $t -and $t -lt (Get-Date).AddMinutes(-2)) {
                # claim the stale lock by an atomic rename; losers of the race just retry
                $line2 = (Get-Content -LiteralPath $f -ErrorAction SilentlyContinue | Select-Object -First 1)
                if ($line2 -eq $line) {
                    $trash = "$($script:NpuLockDir).stale-$([guid]::NewGuid().ToString('N'))"
                    try { [System.IO.Directory]::Move($script:NpuLockDir, $trash); Remove-Item -LiteralPath $trash -Recurse -Force -ErrorAction SilentlyContinue } catch {}
                    continue
                }
            }
        }
        if ((Get-Date) -gt $deadline) { throw "NPU lock timeout (held by: $(Get-Content -LiteralPath $f -ErrorAction SilentlyContinue))" }
        Start-Sleep -Milliseconds 200
    }
}
function Exit-NpuLock {
    if ($env:NPU_LOCK_HELD -eq '1') { return }
    # only release a lock this process owns (a forced takeover must not be undone by the old holder's finally block)
    $f = Join-Path $script:NpuLockDir 'owner.txt'
    $line = (Get-Content -LiteralPath $f -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($line -match "pid=$PID\b") { Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue }
}

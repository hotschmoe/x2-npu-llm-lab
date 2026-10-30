# trace agent: thin wrapper over npu-lock.ps1 (atomic acquire + owner-checked release since ac5e2c8, FIFO tickets since
# 7c47cf6). It never polls the lock itself: every acquisition goes through Enter-NpuLock's ticket queue.
# - an error thrown by Enter-NpuLock is caught and Enter-NpuLock is called again (new ticket), so a caller's
#   ErrorActionPreference=Stop cannot kill a queued run (00:35 on 2026-10-30, pre-ac5e2c8 lock: every waiting run of mine
#   died on a transient owner.txt read);
# - Enter-NpuLock returning means we created the lock dir (atomic); if our owner.txt write did not land, it is rewritten;
# - before running, it waits (holding the lock) while any other llama-*/test-backend-ops process is alive, so a straggler
#   from a run that started without a valid lock (pre-ac5e2c8 collisions at 00:43:10, 00:55:43, 01:08:48) is never
#   overlapped. Normally there is none and this costs nothing.
. (Join-Path $PSScriptRoot 'npu-lock.ps1')
function Get-TraceNpuProcs { Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'llama-*' -or $_.ProcessName -eq 'test-backend-ops' } }
function Test-TraceLockOurs([string]$tag) {
    $f = Join-Path $script:NpuLockDir 'owner.txt'
    for ($i = 0; $i -lt 5; $i++) {
        try { $c = Get-Content -LiteralPath $f -ErrorAction Stop; return [bool]($c -and (@($c)[0] -like "$tag *")) } catch { Start-Sleep -Milliseconds 300 }
    }
    return $false
}
function Enter-TraceLock([string]$Owner) {
    $ErrorActionPreference = 'Continue'
    $tag = "$Owner pid=$PID"
    while ($true) {
        try { Enter-NpuLock $Owner; break } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not (Test-TraceLockOurs $tag)) {
        try { "$tag started=$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $script:NpuLockDir 'owner.txt') -ErrorAction Stop } catch { }
    }
    $warned = $false
    while (Get-TraceNpuProcs) {
        if (-not $warned) { Write-Warning 'lock is ours but another model process is still running: waiting for it to end'; $warned = $true }
        Start-Sleep -Seconds 2
    }
}
function Exit-TraceLock([string]$Owner) {
    if (Test-TraceLockOurs "$Owner pid=$PID") { Exit-NpuLock } else { Write-Warning 'lock no longer ours at exit' }
}

# Same lock protocol as npu-lock.ps1 (atomic mkdir of npu_campaign\.npu.lock, owner.txt), but with a cheap Win32 CreateDirectory poll (~1 ms)
# instead of a 5 s sleep + exception-heavy loop. Needed because a 10 ms python poller (decode2 repeat_server) wins every race against the 5 s poller.
if (-not ('PfLockNative' -as [type])) {
    Add-Type -Namespace '' -Name PfLockNative -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)] public static extern bool CreateDirectoryW(string path, System.IntPtr sa);' -ErrorAction SilentlyContinue
    Add-Type -TypeDefinition 'public static class PfLock { [System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)] public static extern bool CreateDirectoryW(string path, System.IntPtr sa); }' -ErrorAction SilentlyContinue
}
$script:NpuLockDir = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
function Enter-NpuLock {
    param([string]$Owner = 'unknown', [int]$TimeoutSec = 14400)
    if ($env:NPU_LOCK_HELD -eq '1') { return }   # caller (lockrun*.py) already holds the lock
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $lastCheck = Get-Date
    while ($true) {
        if ([PfLock]::CreateDirectoryW($script:NpuLockDir, [System.IntPtr]::Zero)) {
            "$Owner pid=$PID started=$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $script:NpuLockDir 'owner.txt')
            return
        }
        if (((Get-Date) - $lastCheck).TotalSeconds -gt 30) {
            $lastCheck = Get-Date
            $f = Join-Path $script:NpuLockDir 'owner.txt'
            if (Test-Path -LiteralPath $f) {
                $t = (Get-Item -LiteralPath $f).LastWriteTime
                $ownerPid = ((Get-Content -LiteralPath $f) -replace '.*pid=(\d+).*', '$1')
                $alive = $ownerPid -match '^\d+$' -and (Get-Process -Id ([int]$ownerPid) -ErrorAction SilentlyContinue)
                if (-not $alive -and $t -lt (Get-Date).AddMinutes(-2)) { Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue; continue }
            }
            if ((Get-Date) -gt $deadline) { throw "NPU lock timeout (held by: $(Get-Content -LiteralPath $f -ErrorAction SilentlyContinue))" }
        }
        Start-Sleep -Milliseconds 2
    }
}
function Exit-NpuLock { if ($env:NPU_LOCK_HELD -eq '1') { return }; Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue }

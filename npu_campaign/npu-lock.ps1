# Cross-agent NPU mutex: only one inference/benchmark process on the NPU at a time (timings are otherwise garbage).
# Usage (dot-source): . "$PSScriptRoot\npu-lock.ps1"; Enter-NpuLock 'owner-label'; try { ... } finally { Exit-NpuLock }
# Protocol (shared with lockrun*.py, which use os.mkdir): the lock is held by whoever CREATED the directory.
# Acquire uses Win32 CreateDirectoryW, which fails atomically if the directory exists. New-Item / .NET CreateDirectory
# are check-then-create (they succeed on an existing directory), which let several waiters "win" in the same tick.
# Fairness: waiters take a FIFO ticket in <lockdir>.queue (<utc ticks>-<pid>.txt); only the oldest live ticket may try
# to acquire, so poll rate no longer decides who wins. Tickets of dead processes, or with no heartbeat for 30 s, are
# pruned. Do not bypass the queue with a private fast-poll loop.
$script:NpuLockDir = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
if (-not ('NpuLockNative' -as [type])) {
    Add-Type -Namespace '' -Name 'NpuLockNative' -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)]
public static extern bool CreateDirectoryW(string path, System.IntPtr security);
'@
}
function Get-NpuLockOwner {
    # returns @{ pid = <int or $null>; text = <string>; age = <TimeSpan or $null> }; tolerant of a concurrent rewrite
    $f = Join-Path $script:NpuLockDir 'owner.txt'
    $r = @{ pid = $null; text = ''; age = $null }
    try {
        $fs = [IO.File]::Open($f, 'Open', 'Read', 'ReadWrite, Delete')
        try { $r.text = (New-Object IO.StreamReader($fs)).ReadToEnd() } finally { $fs.Dispose() }
        $r.age = (Get-Date) - [IO.File]::GetLastWriteTime($f)
        if ($r.text -match 'pid=(\d+)') { $r.pid = [int]$Matches[1] }
    } catch {
        try { $r.age = (Get-Date) - [IO.Directory]::GetCreationTime($script:NpuLockDir) } catch {}
    }
    return $r
}
function Get-NpuLockHead {
    # oldest live ticket path in the queue (prunes dead / silent tickets on the way)
    $qd = "$($script:NpuLockDir).queue"
    $now = [DateTime]::UtcNow
    foreach ($t in @([IO.Directory]::GetFiles($qd, '*.txt') | Sort-Object)) {
        $name = [IO.Path]::GetFileNameWithoutExtension($t)
        $tpid = if ($name -match '-(\d+)$') { [int]$Matches[1] } else { 0 }
        $dead = $tpid -ne $PID -and -not (Get-Process -Id $tpid -ErrorAction SilentlyContinue)
        $silent = $false
        try { $silent = ($now - [IO.File]::GetLastWriteTimeUtc($t)).TotalSeconds -gt 30 } catch { continue }
        if ($dead -or ($silent -and $tpid -ne $PID)) { try { [IO.File]::Delete($t) } catch {}; continue }
        return $t
    }
    return $null
}
function Enter-NpuLock {
    param([string]$Owner = 'unknown', [int]$TimeoutSec = 14400)
    if ($env:NPU_LOCK_HELD -eq '1') { return }   # caller (lockrun.py) already holds the lock
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $qd = "$($script:NpuLockDir).queue"
    [void][IO.Directory]::CreateDirectory($qd)
    $ticket = Join-Path $qd ('{0:D20}-{1}.txt' -f [DateTime]::UtcNow.Ticks, $PID)
    [IO.File]::WriteAllText($ticket, "$Owner pid=$PID queued=$(Get-Date -Format o)")
    $beat = [Diagnostics.Stopwatch]::StartNew()
    try {
        while ($true) {
            if ($beat.Elapsed.TotalSeconds -gt 2) { try { [IO.File]::SetLastWriteTimeUtc($ticket, [DateTime]::UtcNow) } catch {}; $beat.Restart() }
            if ((Get-NpuLockHead) -eq $ticket) {
                if ([NpuLockNative]::CreateDirectoryW($script:NpuLockDir, [IntPtr]::Zero)) {
                    "$Owner pid=$PID started=$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $script:NpuLockDir 'owner.txt')
                    return
                }
                $o = Get-NpuLockOwner
                $alive = $o.pid -and (Get-Process -Id $o.pid -ErrorAction SilentlyContinue)
                if (-not $alive -and $o.age -and $o.age.TotalMinutes -gt 2) {
                    # stale: owner dead (or never wrote owner.txt) for > 2 min. Only remove it if it is still the same holder.
                    $again = Get-NpuLockOwner
                    if ($again.text -eq $o.text) { Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue }
                    continue
                }
                Start-Sleep -Milliseconds 20
            } else {
                Start-Sleep -Milliseconds 100
            }
            if ((Get-Date) -gt $deadline) { throw "NPU lock timeout (held by: $((Get-NpuLockOwner).text))" }
        }
    } finally {
        try { [IO.File]::Delete($ticket) } catch {}
    }
}
function Exit-NpuLock {
    if ($env:NPU_LOCK_HELD -eq '1') { return }
    $o = Get-NpuLockOwner
    if ($o.pid -eq $PID) { Remove-Item -LiteralPath $script:NpuLockDir -Recurse -Force -ErrorAction SilentlyContinue }
}

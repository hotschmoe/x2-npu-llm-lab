# Mutual-exclusion stress test for npu-lock.ps1 (uses a scratch lock dir via NPU_LOCK_DIR_TEST; never touches the real .npu.lock).
# Usage: test-npu-lock.ps1 [-LockScript path] [-Workers 8] [-Iters 15]   -> prints VIOLATIONS=<n> (must be 0)
# Negative control: run it with the OLD npu-lock.ps1 (git show <old-commit>:npu_campaign/npu-lock.ps1 > old.ps1); the old New-Item based lock is racy
# and is expected to report violations (if it reports 0 over several tries the test is not sensitive enough).
param([string]$LockScript = "$PSScriptRoot\npu-lock.ps1", [int]$Workers = 8, [int]$Iters = 15, [switch]$Worker, [int]$Id = 0)
$ErrorActionPreference = 'Stop'
$testDir = Join-Path $env:TEMP 'npu-lock-test'
if ($Worker) {
    $env:NPU_LOCK_DIR_TEST = "$testDir\lock"
    . $LockScript
    $flag = "$testDir\inside.flag"
    for ($i = 0; $i -lt $Iters; $i++) {
        Enter-NpuLock "w$Id"
        try {
            try { $fs = [System.IO.File]::Open($flag, 'CreateNew', 'Write', 'None') } catch { Add-Content "$testDir\violations.txt" "w$Id iter $i : someone else inside"; $fs = $null }
            Start-Sleep -Milliseconds (Get-Random -Minimum 5 -Maximum 40)
            if ($fs) { $fs.Close(); Remove-Item -LiteralPath $flag -Force -ErrorAction SilentlyContinue }
        } finally { Exit-NpuLock }
    }
    exit 0
}
Remove-Item $testDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $testDir | Out-Null
$ps = @()
for ($w = 0; $w -lt $Workers; $w++) {
    $ps += Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Worker', '-Id', $w, '-Iters', $Iters, '-LockScript', "`"$LockScript`"" -PassThru -WindowStyle Hidden
}
$ps | Wait-Process
$v = if (Test-Path "$testDir\violations.txt") { @(Get-Content "$testDir\violations.txt").Count } else { 0 }
"WORKERS=$Workers ITERS=$Iters VIOLATIONS=$v"
Remove-Item $testDir -Recurse -Force -ErrorAction SilentlyContinue

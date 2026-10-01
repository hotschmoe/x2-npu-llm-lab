# Stop the server started by start-qwen.ps1 (kills only the recorded PIDs) and release the NPU lock. Safe to run when nothing is running.
$runDir = Join-Path $PSScriptRoot 'run'
$root = Split-Path $PSScriptRoot -Parent
. "$root\npu_campaign\npu-lock.ps1"
function Kill-Pid($file, $what) {
    if (-not (Test-Path $file)) { return }
    $p = [int](Get-Content $file)
    if (Get-Process -Id $p -ErrorAction SilentlyContinue) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; Write-Host "stopped $what pid $p" }
}
Kill-Pid (Join-Path $runDir 'server.pid') 'llama-server'
# the supervisor exits by itself once the server process is fully gone (up to ~15 s teardown) and then releases the lock; wait for it, force only after 2 min
$sup = Join-Path $runDir 'supervisor.pid'
$owner = Join-Path $root 'npu_campaign\.npu.lock\owner.txt'
for ($i = 0; $i -lt 120; $i++) {
    if (-not (Test-Path $sup)) { break }
    $sp = [int](Get-Content $sup); if (-not (Get-Process -Id $sp -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Seconds 1
}
Kill-Pid $sup 'supervisor'
Remove-Item (Join-Path $runDir '*.pid'), (Join-Path $runDir 'mode.txt') -ErrorAction SilentlyContinue
if (Test-Path $owner) { Write-Host "NPU lock now held by: $(Get-Content $owner)" } else { Write-Host 'NPU lock free' }

# Robust wrapper around npu-lock.ps1 for the topk scripts. Enter-NpuLock can throw (owner.txt sharing violation / array read)
# AFTER it created the lock dir; that leaves a lock without owner.txt which nobody ever clears. Clean our own half-acquired lock, retry.
. (Join-Path $PSScriptRoot 'npu-lock.ps1')
function Enter-TopkLock([string]$Owner) {
    $dir = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
    for ($i = 0; $i -lt 2000; $i++) {
        try { Enter-NpuLock $Owner; return } catch {
            "$(Get-Date -Format o) lock retry: $($_.Exception.Message)" | Add-Content (Join-Path $PSScriptRoot 'clover-logs\topk-lockerr.txt')
            try {
                if ((Test-Path -LiteralPath $dir) -and -not (Test-Path -LiteralPath "$dir\owner.txt") -and
                    ((Get-Date) - (Get-Item -LiteralPath $dir).CreationTime).TotalSeconds -gt 5) {
                    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
                }
            } catch {}
            Start-Sleep -Seconds 2
        }
    }
    throw 'topk lock: gave up'
}
function Remove-HalfLock {
    $dir = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
    try {
        if ((Test-Path -LiteralPath $dir) -and -not (Test-Path -LiteralPath "$dir\owner.txt") -and
            ((Get-Date) - (Get-Item -LiteralPath $dir).CreationTime).TotalSeconds -gt 5) { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue }
    } catch {}
}

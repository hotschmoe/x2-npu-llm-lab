# Run a sequence of Clover draft-self runs under ONE NPU lock hold (keep <= ~10 min: 4 runs of 256 tokens).
# Arms: array of @{n=..;pkg=..;env=@{}} in the order to run (e.g. A B B A). Prints tok/s, host cores and an output hash per run.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][object[]]$Seq,
    [string]$Prompt = 'npu_campaign\clover-prompts\diffusion.txt',
    [int]$NPredict = 256
)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. "$repo\npu_campaign\npu-lock.ps1"
Enter-NpuLock "blk.ps1 $Tag"
$i = 0
try {
    $env:NPU_LOCK_HELD = '1'   # clover-run.ps1 then skips its own lock
    foreach ($a in $Seq) {
        $i++
        $label = "$Tag-$i-$($a.n)"
        $before = @(Get-Process -Name 'llama-*', 'test-backend-ops' -ErrorAction SilentlyContinue).Count
        $envBackup = @{}
        foreach ($k in $a.env.Keys) { $envBackup[$k] = [Environment]::GetEnvironmentVariable($k); Set-Item "env:$k" $a.env[$k] }
        $out = & "$repo\npu_campaign\clover-run.ps1" -Label $label -Pkg $a.pkg -Accept '' -NPredict $NPredict -Prompt $Prompt -Extra $a.extra 2>&1 | Out-String
        foreach ($k in $envBackup.Keys) { if ($null -eq $envBackup[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue } else { Set-Item "env:$k" $envBackup[$k] } }
        $after = @(Get-Process -Name 'llama-*', 'test-backend-ops' -ErrorAction SilentlyContinue).Count
        $tps = $null; $cpu = $null
        $m = Select-String -Path "$repo\npu_campaign\clover-logs\$label.err" -Pattern 'decoded +(\d+) tokens in +([\d.]+) seconds, speed: +([\d.]+)' | Select-Object -Last 1
        if ($m) { $tps = [double]$m.Matches[0].Groups[3].Value }
        if ($out -match 'HOST_CPU busy cores during generation = ([\d.]+)') { $cpu = [double]$Matches[1] }
        $h = (Get-FileHash "$repo\npu_campaign\clover-logs\$label.log" -Algorithm SHA256).Hash.Substring(0, 10)
        "{0} #{1} {2}: {3} tok/s, host cores {4}, out-hash {5}, other llama procs before/after {6}/{7}" -f $Tag, $i, $a.n, $tps, $cpu, $h, $before, $after
    }
} finally {
    Remove-Item env:NPU_LOCK_HELD -ErrorAction SilentlyContinue
    Exit-NpuLock
}

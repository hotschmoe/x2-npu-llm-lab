# Clover draft-self runs in ONE process that holds the NPU lock and starts every exe itself (no unlocked children):
# Enter-NpuLock -> for each arm Start-Process + WaitForExit(timeout) -> finally kill any live child, Exit-NpuLock.
# Keep <= ~10 min per call (4 runs of 256 tokens). Arms: @{n=..;pkg=..;env=@{}} in run order (e.g. A B B A).
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][object[]]$Seq,
    [string]$Prompt = 'npu_campaign\clover-prompts\diffusion.txt',
    [int]$NPredict = 256,
    [int]$RunTimeoutSec = 420
)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$model = "$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf"
. "$repo\npu_campaign\npu-lock.ps1"
Enter-NpuLock "blk2.ps1 $Tag"
$run = $null
try {
    $i = 0
    foreach ($a in $Seq) {
        $i++
        $label = "$Tag-$i-$($a.n)"
        $pkgDir = (Resolve-Path (Join-Path $repo $a.pkg)).Path
        $env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
        foreach ($k in $a.env.Keys) { Set-Item "env:$k" $a.env[$k] }
        $args_ = "-m `"$model`" -f `"$repo\$Prompt`" --spec-type draft-self --spec-draft-n-max 15 --temp 0 --top-k 1 --seed 42 -ctk f16 -ctv f16 " +
                 "-c 4096 -n $NPredict -b 256 -ub 256 -ngl 99 --device HTP0 -t 8 -tb 8 --flash-attn on --color off $($a.extra)"
        $out = "$repo\npu_campaign\clover-logs\$label"
        $before = @(Get-Process -Name 'llama-*', 'test-backend-ops' -ErrorAction SilentlyContinue).Count
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $samples = [System.Collections.Generic.List[object]]::new()
        $run = Start-Process -FilePath "$pkgDir\bin\llama-speculative-simple.exe" -ArgumentList $args_ -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
        $null = $run.Handle
        while (-not $run.HasExited -and $sw.Elapsed.TotalSeconds -lt $RunTimeoutSec) {
            try { $samples.Add([pscustomobject]@{ t = $sw.Elapsed.TotalSeconds; cpu = $run.TotalProcessorTime.TotalSeconds }) } catch {}
            Start-Sleep -Milliseconds 250
        }
        $timedOut = -not $run.HasExited
        if ($timedOut) { Stop-Process -Id $run.Id -Force -ErrorAction SilentlyContinue; $run.WaitForExit(30000) | Out-Null }
        foreach ($k in $a.env.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
        $tps = $null; $cores = $null
        $m = Select-String -Path "$out.err" -Pattern 'decoded +(\d+) tokens in +([\d.]+) seconds, speed: +([\d.]+)' | Select-Object -Last 1
        if ($m) {
            $tps = [double]$m.Matches[0].Groups[3].Value; $gen = [double]$m.Matches[0].Groups[2].Value
            if ($samples.Count -gt 8) {
                $tEnd = $samples[$samples.Count - 1].t - 0.5; $tBeg = $tEnd - $gen
                $w = @($samples | Where-Object { $_.t -ge $tBeg -and $_.t -le $tEnd })
                if ($w.Count -ge 2) { $cores = ($w[-1].cpu - $w[0].cpu) / ($w[-1].t - $w[0].t) }
            }
        }
        $h = if (Test-Path "$out.log") { (Get-FileHash "$out.log" -Algorithm SHA256).Hash.Substring(0, 10) } else { 'none' }
        "{0} #{1} {2}: {3} tok/s, host cores {4:F2}, out-hash {5}, timeout {6}, other llama procs before {7}" -f $Tag, $i, $a.n, $tps, $cores, $h, $timedOut, $before
    }
} finally {
    if ($run -and -not $run.HasExited) { Stop-Process -Id $run.Id -Force -ErrorAction SilentlyContinue }
    Exit-NpuLock
}

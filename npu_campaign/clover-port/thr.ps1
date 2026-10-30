# Which host threads burn CPU during a Clover HTP0 run? Samples per-thread CPU time every 250 ms and prints the busiest threads.
# Usage: & thr.ps1 -Label x -Pkg npu_campaign\pkg-clover-port-all [-Env @{...}] [-NPredict 96]
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-port-all',
    [int]$NPredict = 96,
    [hashtable]$Env = @{},
    [string]$Threads = '-t 8 -tb 8',
    [switch]$Qwen
)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item "env:$k" $Env[$k] }
$model = "$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf"
$a = "-m `"$model`" -f `"$repo\npu_campaign\clover-prompts\diffusion.txt`" --spec-type draft-self --spec-draft-n-max 15 --temp 0 --top-k 1 --seed 42 -ctk f16 -ctv f16 " +
     "-c 4096 -n $NPredict -b 256 -ub 256 -ngl 99 --device HTP0 $Threads --flash-attn on --color off"
$exe = 'llama-speculative-simple.exe'
if ($Qwen) {
    $exe = 'llama-completion.exe'
    $a = "-m `"$repo\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`" -p `"Write a long story about a dragon.`" --temp 0 --top-k 1 --seed 42 -n $NPredict -c 2048 -b 256 -ub 256 -ngl 99 --device HTP0 $Threads --flash-attn on --color off -no-cnv"
}
. "$repo\npu_campaign\npu-lock.ps1"; Enter-NpuLock "thr.ps1 $Label"
$prev = @{}; $acc = @{}; $meta = @{}
$sw = [Diagnostics.Stopwatch]::StartNew(); $t_first = $null; $genStart = $null
try {
    $run = Start-Process -FilePath "$pkgDir\bin\$exe" -ArgumentList $a -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$repo\npu_campaign\clover-logs\thr-$Label.log" -RedirectStandardError "$repo\npu_campaign\clover-logs\thr-$Label.err"
    $null = $run.Handle
    $nsamp = 0
    while (-not $run.HasExited) {
        try {
            $run.Refresh()
            # only sample once generation started (first 'speculative' progress is not visible; use the last 60% of the run by recording all and printing deltas after t>=60 s of CPU)
            foreach ($t in $run.Threads) {
                $id = $t.Id
                $cur = $t.TotalProcessorTime.TotalSeconds
                $key = "$id"
                if (-not $meta.ContainsKey($key)) { $meta[$key] = ('0x{0:X}' -f [int64]$t.StartAddress); $acc[$key] = @() }
                $acc[$key] += , @($sw.Elapsed.TotalSeconds, $cur)
            }
            $nsamp++
        } catch {}
        Start-Sleep -Milliseconds 250
    }
} finally { Exit-NpuLock }
$total = $sw.Elapsed.TotalSeconds
$dec = Select-String -Path "$repo\npu_campaign\clover-logs\thr-$Label.err" -Pattern 'decoded +\d+ tokens in +([\d.]+) seconds, speed: +([\d.]+)' | Select-Object -Last 1
$genSec = if ($dec) { [double]$dec.Matches[0].Groups[1].Value } else { 10.0 }
$tps = if ($dec) { $dec.Matches[0].Groups[2].Value } else { '?' }
$lastT = ($acc.Values | ForEach-Object { $_[-1][0] } | Measure-Object -Maximum).Maximum
$t1 = $lastT - 0.5; $t0 = $t1 - $genSec
"{0}: {1} tok/s, generation window {2:F1}s" -f $Label, $tps, $genSec
"WALL {0:F1}s, samples {1}" -f $total, $nsamp
$rows = foreach ($k in $acc.Keys) {
    $s = $acc[$k] | Where-Object { $_[0] -ge $t0 -and $_[0] -le $t1 }
    if ($s.Count -ge 2) { [pscustomobject]@{ tid = $k; start = $meta[$k]; cores = ($s[-1][1] - $s[0][1]) / ($s[-1][0] - $s[0][0]) } }
}
$rows | Sort-Object cores -Descending | Select-Object -First 12 | Format-Table -AutoSize | Out-String
"sum cores in window: {0:F2}" -f (($rows | Measure-Object cores -Sum).Sum)

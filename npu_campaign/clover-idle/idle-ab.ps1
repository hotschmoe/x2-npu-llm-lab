# A/B of host wait strategies under ONE NPU lock hold. Each arm = one process run; reports tok/s, host busy cores over the
# generation window (process CPU time sampled every 250 ms, like clover-run.ps1), the busiest thread's share, and the stdout hash.
# Usage: idle-ab.ps1 -Label ab1 -Arms @(@{n='base'; pkg='...'; env=@{}}, @{n='w1'; pkg='...'; env=@{GGML_HEXAGON_WAIT='1'}}) [-Clover] [-NPredict 600]
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][object[]]$Arms,
    [switch]$Clover,
    [int]$NPredict = 0,
    [string]$Threads = '-t 8 -tb 8',
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$logDir = "$repo\npu_campaign\clover-logs"
if ($Clover) {
    if ($NPredict -le 0) { $NPredict = 256 }
    $exe = 'llama-speculative-simple.exe'
    $a = "-m `"$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf`" -f `"$repo\npu_campaign\clover-prompts\diffusion.txt`" --spec-type draft-self --spec-draft-n-max 15 --temp 0 --top-k 1 --seed 42 -ctk f16 -ctv f16 " +
         "-c 4096 -n $NPredict -b 256 -ub 256 -ngl 99 --device HTP0 $Threads --flash-attn on --color off $Extra"
} else {
    if ($NPredict -le 0) { $NPredict = 600 }
    $exe = 'llama-completion.exe'
    $a = "-m `"$repo\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`" -p `"Write a long story about a dragon.`" --temp 0 --top-k 1 --seed 42 -n $NPredict -c 2048 -b 256 -ub 256 -ngl 99 --device HTP0 $Threads --flash-attn on --color off -no-cnv $Extra"
}
$results = @()
. "$repo\npu_campaign\npu-lock.ps1"; Enter-NpuLock "idle ab $Label"
try {
    foreach ($arm in $Arms) {
        $name = $arm.n
        $pkgDir = (Resolve-Path (Join-Path $repo $arm.pkg)).Path
        $saved = @{}
        $env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
        foreach ($k in $arm.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); Set-Item "env:$k" $arm.env[$k] }
        $out = "$logDir\ab-$Label-$name"
        $samples = [System.Collections.Generic.List[object]]::new()
        $thr = @{}
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $run = Start-Process -FilePath "$pkgDir\bin\$exe" -ArgumentList "$a $($arm.extra)" -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
        $null = $run.Handle
        try {
            while (-not $run.HasExited) {
                try {
                    $run.Refresh()
                    $t = $sw.Elapsed.TotalSeconds
                    $samples.Add([pscustomobject]@{ t = $t; cpu = $run.TotalProcessorTime.TotalSeconds })
                    foreach ($th in $run.Threads) {
                        if (-not $thr.ContainsKey($th.Id)) { $thr[$th.Id] = [System.Collections.Generic.List[object]]::new() }
                        $thr[$th.Id].Add(@($t, $th.TotalProcessorTime.TotalSeconds, $th.PrivilegedProcessorTime.TotalSeconds))
                    }
                } catch {}
                Start-Sleep -Milliseconds 250
                if ($sw.Elapsed.TotalSeconds -gt 900) { $run.Kill(); break }
                # skip the slow teardown (buffer release, ~1 core for tens of seconds) unless the arm asks to keep it
                if (-not $arm.keep -and (Select-String -Path "$out.err" -Pattern 'common_perf_print: +graph reallocs|n_accept +=' -Quiet)) { Start-Sleep -Milliseconds 700; try { $run.Kill() } catch {}; break }
            }
            $run.WaitForExit()
        } finally {
            if (-not $run.HasExited) { try { $run.Kill() } catch {} }
            foreach ($k in $saved.Keys) { if ($null -eq $saved[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue } else { Set-Item "env:$k" $saved[$k] } }
        }
        $err = Get-Content "$out.err" -Raw
        $tps = $null; $gen = 0
        if ($err -match 'eval time = +([\d.]+) ms / +(\d+) runs.*?([\d.]+) tokens per second\)') { $gen = [double]$Matches[1] / 1000; $tps = [double]$Matches[3] }
        if ($err -match 'decoded +(\d+) tokens in +([\d.]+) seconds, speed: +([\d.]+)') { $gen = [double]$Matches[2]; $tps = [double]$Matches[3] }
        $cores = $null; $top = $null; $topk = $null; $trows = @()
        if ($gen -gt 1 -and $samples.Count -gt 8) {
            # window from the stderr timestamp (m.ss.mmm.uuu since process start) of the perf line: teardown after generation can take
            # tens of seconds at ~1 core (buffer release), so "last sample - gen" measures teardown, not generation.
            $tsl = Select-String -Path "$out.err" -Pattern '^(\d+)\.(\d+)\.(\d+)\.(\d+) I .*(eval time = .*runs|decoded +\d+ tokens in)' | Select-Object -Last 1
            $tsEnd = [double]$tsl.Matches[0].Groups[1].Value * 60 + [double]$tsl.Matches[0].Groups[2].Value + [double]$tsl.Matches[0].Groups[3].Value / 1e3
            $tEnd = $tStart + $tsEnd; $tBeg = $tEnd - $gen
            $teardown = $samples[$samples.Count - 1].t - $tEnd
            $w = @($samples | Where-Object { $_.t -ge $tBeg -and $_.t -le $tEnd })
            if ($w.Count -ge 2) { $cores = ($w[-1].cpu - $w[0].cpu) / ($w[-1].t - $w[0].t) }
            foreach ($id in $thr.Keys) {
                $s = @($thr[$id] | Where-Object { $_[0] -ge $tBeg -and $_[0] -le $tEnd })
                if ($s.Count -ge 2) {
                    $c = ($s[-1][1] - $s[0][1]) / ($s[-1][0] - $s[0][0]); $k = ($s[-1][2] - $s[0][2]) / ($s[-1][0] - $s[0][0])
                    if ($null -eq $top -or $c -gt $top) { $top = $c; $topk = $k }
                    $trows += , @($id, $c, $k)
                }
            }
        }
        $hash = (Get-FileHash "$out.log").Hash.Substring(0, 12)
        $stats = ([regex]::Matches($err, 'wait-stats: .*') | Select-Object -First 1).Value
        $r = [pscustomobject]@{ arm = $name; tps = $tps; gen_s = [math]::Round($gen, 2); cores = if ($null -ne $cores) { [math]::Round($cores, 3) } else { $null }
                                top_thr = if ($null -ne $top) { [math]::Round($top, 3) } else { $null }; top_kernel = if ($null -ne $topk) { [math]::Round($topk, 3) } else { $null }; hash = $hash; exit = $run.ExitCode; teardown_s = [math]::Round($teardown, 1) }
        $results += $r
        "{0,-10} tok/s {1,7} gen {2,6}s cores {3,6} top-thread {4,6} (kernel {5,6}) hash {6} exit {7} teardown {8}s" -f $r.arm, $r.tps, $r.gen_s, $r.cores, $r.top_thr, $r.top_kernel, $r.hash, $r.exit, $r.teardown_s
        if ($stats) { "           $stats" }
        "           threads: " + (($trows | Sort-Object { - $_[1] } | Select-Object -First 9 | ForEach-Object { "{0}:{1:F3}/k{2:F3}" -f $_[0], $_[1], $_[2] }) -join ' ')
    }
} finally { Exit-NpuLock }
$results | ConvertTo-Json -Compress | Set-Content "$logDir\ab-$Label.json"




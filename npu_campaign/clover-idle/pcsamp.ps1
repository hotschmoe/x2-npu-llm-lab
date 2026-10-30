# Where does the busy host thread spend its time during HTP0 generation? Per-thread census + PC/frame-chain sampling.
# Usage: pcsamp.ps1 -Label base -Pkg npu_campaign\pkg-clover-idle-base [-Clover] [-Env @{...}] [-SampleSec 8] [-ThreadId n]
# Starts the run under the NPU lock, waits until generation is under way (Qwen: 'generate:' in stderr; Clover: StartDelaySec after
# the first stdout bytes), takes a per-thread CPU census (user/kernel, start address), samples the busiest thread every IntervalMs
# for SampleSec, symbolizes (exports + PDBs on SymPath) and prints module / function histograms.
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-idle-base',
    [switch]$Clover,
    [int]$NPredict = 0,
    [hashtable]$Env = @{},
    [double]$SampleSec = 8,
    [double]$IntervalMs = 2,
    [int]$Depth = 12,
    [string]$SymPath = '',
    [string]$Threads = '-t 8 -tb 8',
    [double]$StartDelaySec = 3,
    [double]$CensusSec = 3,
    [int]$ThreadId = 0,
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
if (-not ('PcSampler' -as [type])) { Add-Type -Path "$PSScriptRoot\PcSampler.cs" }
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item "env:$k" $Env[$k] }
if ($Clover) {
    if ($NPredict -le 0) { $NPredict = 256 }
    $exe = 'llama-speculative-simple.exe'
    $a = "-m `"$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf`" -f `"$repo\npu_campaign\clover-prompts\diffusion.txt`" --spec-type draft-self --spec-draft-n-max 15 --temp 0 --top-k 1 --seed 42 -ctk f16 -ctv f16 " +
         "-c 4096 -n $NPredict -b 256 -ub 256 -ngl 99 --device HTP0 $Threads --flash-attn on --color off $Extra"
} else {
    if ($NPredict -le 0) { $NPredict = 900 }
    $exe = 'llama-completion.exe'
    $a = "-m `"$repo\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`" -p `"Write a long story about a dragon.`" --temp 0 --top-k 1 --seed 42 -n $NPredict -c 2048 -b 256 -ub 256 -ngl 99 --device HTP0 $Threads --flash-attn on --color off -no-cnv $Extra"
}
$logDir = "$repo\npu_campaign\clover-logs"
$out = "$logDir\pcs-$Label"
. "$repo\npu_campaign\npu-lock.ps1"; Enter-NpuLock "idle pcsamp $Label"
$stacks = $null; $sym = @{}; $tid = 0; $mods = @()
try {
    $run = Start-Process -FilePath "$pkgDir\bin\$exe" -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    $null = $run.Handle
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not $run.HasExited -and $sw.Elapsed.TotalSeconds -lt 400) {
        if ($Clover) { if ((Get-Item "$out.log").Length -gt 200) { break } }
        else { if (Select-String -Path "$out.err" -Pattern 'generate: n_ctx' -Quiet) { break } }
        Start-Sleep -Milliseconds 200
    }
    "generation trigger at {0:F1}s" -f $sw.Elapsed.TotalSeconds
    Start-Sleep -Milliseconds ([int]($StartDelaySec * 1000))
    if ($run.HasExited) { throw 'process exited before sampling' }
    $snap = {
        $run.Refresh(); $h = @{}
        foreach ($t in $run.Threads) { try { $h[$t.Id] = @($t.TotalProcessorTime.TotalSeconds, $t.PrivilegedProcessorTime.TotalSeconds, [int64]$t.StartAddress) } catch {} }
        $h
    }
    $mods = @($run.Modules | ForEach-Object { [pscustomobject]@{ name = $_.ModuleName; base = [uint64]$_.BaseAddress.ToInt64(); end = [uint64]$_.BaseAddress.ToInt64() + [uint64]$_.ModuleMemorySize } })
    $c0 = & $snap; $cw = [Diagnostics.Stopwatch]::StartNew(); Start-Sleep -Milliseconds ([int]($CensusSec * 1000)); $c1 = & $snap; $dt = $cw.Elapsed.TotalSeconds
    $rows = foreach ($id in $c1.Keys) {
        $sa = [uint64]$c1[$id][2]; $where = '0x{0:x}' -f $sa
        foreach ($m in $mods) { if ($sa -ge $m.base -and $sa -lt $m.end) { $where = '{0}+0x{1:x}' -f $m.name, ($sa - $m.base); break } }
        if ($c0.ContainsKey($id)) { [pscustomobject]@{ tid = $id; cores = [math]::Round(($c1[$id][0] - $c0[$id][0]) / $dt, 3); kernel = [math]::Round(($c1[$id][1] - $c0[$id][1]) / $dt, 3); start = $where } }
        else { [pscustomobject]@{ tid = $id; cores = -1; kernel = -1; start = "new $where" } }
    }
    "census over {0:F1}s, {1} threads (top 8 by cores):" -f $dt, $c1.Count
    ($rows | Sort-Object cores -Descending | Select-Object -First 8 | Format-Table -AutoSize | Out-String -Width 200).TrimEnd()
    $tid = if ($ThreadId -gt 0) { $ThreadId } else { ($rows | Sort-Object cores -Descending | Select-Object -First 1).tid }
    "sampling thread $tid"
    $stacks = [PcSampler]::Sample($run.Id, $tid, [int]($SampleSec * 1000), $IntervalMs, $Depth)
    "sampler: $([PcSampler]::Info) wall {0:F2}s kernel {1:F2}s user {2:F2}s cycles {3:F2}G" -f [PcSampler]::WallSec, ([PcSampler]::KernelTicks / 1e7), ([PcSampler]::UserTicks / 1e7), ([PcSampler]::Cycles / 1e9)
    $addrs = New-Object 'System.Collections.Generic.HashSet[uint64]'
    foreach ($s in $stacks) { foreach ($x in $s) { [void]$addrs.Add($x) } }
    $sp = if ($SymPath) { $SymPath } else { "$pkgDir\bin" }
    $sym = [PcSampler]::Symbolize($run.Id, $addrs, $sp)
    if (-not $run.HasExited) { $null = $run.WaitForExit(300000) }
} finally {
    if ($run -and -not $run.HasExited) { try { $run.Kill() } catch {} }
    Exit-NpuLock
}
function ModOf([uint64]$x) { foreach ($m in $mods) { if ($x -ge $m.base -and $x -lt $m.end) { return $m.name } }; return '?' }
function Name([uint64]$x) { $m = ModOf $x; $f = if ($sym.ContainsKey($x)) { $sym[$x] } else { '0x{0:x}' -f ($x) }; return "$m!$f" }
$n = $stacks.Count
"--- top-of-stack module (PC), n=$n"
$stacks | Group-Object { ModOf $_[0] } | Sort-Object Count -Descending | Select-Object -First 10 | ForEach-Object { "{0,6:P1}  {1}" -f ($_.Count / $n), $_.Name }
"--- top-of-stack function (PC)"
$stacks | Group-Object { (Name $_[0]) -replace '\+0x[0-9a-f]+$', '' } | Sort-Object Count -Descending | Select-Object -First 15 | ForEach-Object { "{0,6:P1}  {1}" -f ($_.Count / $n), $_.Name }
"--- first frame outside ntdll/KERNELBASE/kernel32 (LR + FP chain)"
$stacks | Group-Object {
    $r = 'none'; foreach ($x in $_) { $m = ModOf $x; if ($m -notin 'ntdll.dll', 'KERNELBASE.dll', 'KERNEL32.DLL', '?') { $r = (Name $x) -replace '\+0x[0-9a-f]+$', ''; break } }; $r
} | Sort-Object Count -Descending | Select-Object -First 15 | ForEach-Object { "{0,6:P1}  {1}" -f ($_.Count / $n), $_.Name }
"--- 5 most common stacks (8 frames)"
$stacks | Group-Object { ($_ | Select-Object -First 8 | ForEach-Object { Name $_ }) -join ' < ' } | Sort-Object Count -Descending | Select-Object -First 5 | ForEach-Object { "{0,6:P1}  {1}" -f ($_.Count / $n), $_.Name }
Select-String -Path "$out.err" -Pattern 'eval time = .*tokens per second\)|decoded +\d+ tokens in.*' | Select-Object -Last 2 | ForEach-Object { $_.Line.Trim() }

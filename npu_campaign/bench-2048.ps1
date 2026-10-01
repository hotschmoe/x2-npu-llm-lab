# Quick pp/tg benchmark at 2048 context for the Qwen3.6-35B-A3B Q4_0 model on HTP0: pp2048 (prompt processing), tg128 at depth 0 and at depth 2048.
# Interleaved A,B,A,B between two packages (default: installed pkg-wos vs previous build pkg-final2-f1). Takes the NPU lock per run. AC power only.
# Usage: powershell -File bench-2048.ps1 [-Rounds 2] [-Reps 2] [-Pkgs 'llama.cpp-hexagon\pkg-wos','npu_campaign\pkg-final2-f1']
param([int]$Rounds = 2, [int]$Reps = 2, [string[]]$Pkgs = @('llama.cpp-hexagon\pkg-wos', 'npu_campaign\pkg-final2-f1'), [string]$Model = 'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf', [int]$UB = 1024)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not (Get-CimInstance -Namespace root\WMI -ClassName BatteryStatus).PowerOnline) { throw 'on battery: timings invalid' }
. (Join-Path $PSScriptRoot 'npu-lock.ps1')
$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($pkg in $Pkgs) {
        $dir = Join-Path $repoRoot $pkg
        $env:GGML_HEXAGON_OPBATCH = '3072'; $env:GGML_HEXAGON_REPLAY = '1'; $env:GGML_HEXAGON_OPFUSION = '1'; $env:GGML_HEXAGON_ROUTER_HVX = '1'
        $env:GGML_HEXAGON_MM_SELECT = '2'; $env:GGML_HEXAGON_FA_SELECT = '2'; $env:GGML_HEXAGON_GDN_SELECT = '2'
        $env:ADSP_LIBRARY_PATH = "$dir\lib;"
        $argv = @('-m', "$repoRoot\$Model", '-p', '2048', '-n', '128', '-d', '0,2048', '-b', '2048', '-ub', "$UB", '-r', "$Reps", '-ngl', '99', '--device', 'HTP0', '-o', 'md')
        Enter-NpuLock "bench-2048 $pkg"
        try {
            $ErrorActionPreference = 'Continue'   # llama-bench prints a driver banner on stderr; do not turn it into a terminating error
            $out = & "$dir\bin\llama-bench.exe" @argv 2>$null
            $ErrorActionPreference = 'Stop'
        } finally { Exit-NpuLock }
        foreach ($l in $out) { if ($l -match '\|\s*(pp\d+|tg\d+)(?:\s*@\s*d(\d+))?\s*\|\s*([\d.]+)') { $rows += [pscustomobject]@{ Pkg = (Split-Path $pkg -Leaf); Round = $r; Test = "$($Matches[1])" + $(if ($Matches[2]) { "@d$($Matches[2])" } else { '' }); TPS = [double]$Matches[3] } } }
    }
}
$rows | Group-Object Pkg, Test | ForEach-Object { $v = $_.Group | Measure-Object TPS -Average -Minimum -Maximum; [pscustomobject]@{ Pkg = $_.Group[0].Pkg; Test = $_.Group[0].Test; Mean = [math]::Round($v.Average, 1); Min = [math]::Round($v.Minimum, 1); Max = [math]::Round($v.Maximum, 1); N = $v.Count } } | Sort-Object Test, Pkg | Format-Table -AutoSize

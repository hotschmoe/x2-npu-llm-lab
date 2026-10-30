# Clover causal-path correctness gate: KLD of a package's logits vs the CPU reference logits (always takes the NPU lock).
# Usage: clover-kld.ps1 -Label bf16-v1 -Pkg npu_campaign\pkg-clover-bf16-v1 [-Device HTP0] [-Env @{...}]
# Reference (default, since 2026-10-30, trace agent): cpu-f32dense-kld-base.bin = CPU run of the Q4_0 GGUF with the 311 dense
# tensors dequantized to F32 (experts unchanged), PPL 15.125. The plain CPU Q4_0 run (cpu-kld-base.bin, PPL 15.4413) is
# LESS accurate than HTP0 (ggml-cpu quantizes activations to Q8_0; Clover activations have absmax/rms up to 63).
# Gate: mean KLD <= 0.025 and same top >= 92%. Baseline HTP0 (pkg-clover-base / M0): KLD 0.0199, same top 93.1%,
# PPL 15.083 (noise floor ~0.018: one HTP build vs itself with an exact router gives 0.0177).
# Old reference for comparisons: -Base npu_campaign\clover-logs\cpu-kld-base.bin (M0 HTP0: 0.0479 / 89.2%).
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-base',
    [string]$Device = 'HTP0',
    [string]$Model = 'models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf',
    [string]$Base = 'npu_campaign\clover-logs\cpu-f32dense-kld-base.bin',
    [int]$Chunks = 4,
    [hashtable]$Env = @{}
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid label' }
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$ngl = if ($Device -eq 'none') { 0 } else { 99 }
$a = "-m `"$repo\$Model`" -f `"$repo\npu_campaign\ppl_text.txt`" -c 512 -b 512 -ub 512 --chunks $Chunks " +
     "-ngl $ngl --device $Device -t 12 -tb 12 -fit off -ctk f16 -ctv f16 --kl-divergence-base `"$repo\$Base`" --kl-divergence"
$out = "$PSScriptRoot\clover-logs\kld-$Label"
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "clover-kld.ps1 $Label"
try {
    $run = Start-Process -FilePath "$pkgDir\bin\llama-perplexity.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
} finally { Exit-NpuLock }
"EXIT=$($run.ExitCode)"
Select-String -Path "$out.log", "$out.err" -Pattern 'Mean PPL\(Q\) |Mean PPL\(base\)|Mean    KLD|Same top|RMS' | ForEach-Object { $_.Line }
$kld = Select-String -Path "$out.log", "$out.err" -Pattern 'Mean    KLD: +([\d.]+)' | Select-Object -Last 1
$top = Select-String -Path "$out.log", "$out.err" -Pattern 'Same top p: +([\d.]+)' | Select-Object -Last 1
if ($kld -and $top -and $Base -like '*f32dense*') {
    $k = [double]$kld.Matches[0].Groups[1].Value; $t = [double]$top.Matches[0].Groups[1].Value
    "GATE " + $(if ($k -le 0.025 -and $t -ge 92.0) { 'PASS' } else { 'FAIL' }) + " (KLD $k <= 0.025, same top $t >= 92)"
}

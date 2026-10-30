# Clover causal-path correctness gate: KLD of a package's logits vs the CPU reference logits (always takes the NPU lock).
# Usage: clover-kld.ps1 -Label bf16-v1 -Pkg npu_campaign\pkg-clover-bf16-v1 [-Device HTP0] [-Env @{...}]
# Reference: npu_campaign\clover-logs\cpu-kld-base.bin (pkg-clover-base, CPU, f16 KV, ppl_text.txt 4x512, PPL 15.4413).
# Baseline HTP0 (pkg-clover-base): mean KLD 0.0479, same top 89.2%, PPL 15.0832.
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Pkg = 'npu_campaign\pkg-clover-base',
    [string]$Device = 'HTP0',
    [string]$Model = 'models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf',
    [string]$Base = 'npu_campaign\clover-logs\cpu-kld-base.bin',
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

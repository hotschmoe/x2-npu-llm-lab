# Capture real Clover router logits (ffn_moe_logits-*) from a CPU run of llama-eval-callback (holds the NPU lock: DRAM bandwidth).
param([string]$Pkg = 'npu_campaign\pkg-clover-topk-a', [string]$Out = 'npu_campaign\clover-logs\topk-capture', [int]$Ub = 512)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
New-Item -ItemType Directory -Force "$repo\$Out" | Out-Null
Remove-Item "$repo\$Out\*.f32" -ErrorAction SilentlyContinue
$env:COMMON_DEBUG_DUMP_DIR = "$repo\$Out"
$env:COMMON_DEBUG_DUMP_RE = '^ffn_moe_logits-\d+$'
. (Join-Path $PSScriptRoot 'topk-lockutil.ps1'); Enter-TopkLock 'topk-capture.ps1 (CPU eval-callback)'
try {
    $a = "-m `"$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf`" -f `"$repo\npu_campaign\clover-prompts\topk-capture.txt`" -ngl 0 --device none -t 12 -c 4096 -b 4096 -ub $Ub --seed 42"
    $p = Start-Process -FilePath "$pkgDir\bin\llama-eval-callback.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$repo\$Out\capture.log" -RedirectStandardError "$repo\$Out\capture.err"
} finally { Exit-NpuLock }
"EXIT=$($p.ExitCode)"
Get-ChildItem "$repo\$Out\*.f32" | Measure-Object Length -Sum | ForEach-Object { "files=$($_.Count) bytes=$($_.Sum)" }

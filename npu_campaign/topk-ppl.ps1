# llama-perplexity from a pkg under the NPU lock. -Mode cpu: CPU PPL (expect 15.4413). -Mode kld: KLD vs clover-logs\cpu-kld-base.bin on HTP0.
param([Parameter(Mandatory=$true)][string]$Label, [string]$Pkg = 'npu_campaign\pkg-clover-topk-a', [ValidateSet('cpu','kld')][string]$Mode = 'cpu',
      [hashtable]$Env = @{})
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$pkgDir = (Resolve-Path (Join-Path $repo $Pkg)).Path
$env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$logDir = "$PSScriptRoot\clover-logs"
$base = "-m `"$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf`" -f `"$repo\npu_campaign\ppl_text.txt`" -c 512 -b 512 -ub 512 --chunks 4"
$a = if ($Mode -eq 'cpu') { "$base -ngl 0 --device none -t 12" } else { "$base --kl-divergence-base `"$logDir\cpu-kld-base.bin`" --kl-divergence -ngl 99 --device HTP0 -fit off" }
. (Join-Path $PSScriptRoot 'topk-lockutil.ps1'); Enter-TopkLock "topk-ppl.ps1 $Label"
try {
    $p = Start-Process -FilePath "$pkgDir\bin\llama-perplexity.exe" -ArgumentList $a -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$logDir\$Label.log" -RedirectStandardError "$logDir\$Label.err"
} finally { Exit-NpuLock }
"EXIT=$($p.ExitCode)"
Select-String -Path "$logDir\$Label.err","$logDir\$Label.log" -Pattern 'Final estimate|Mean +KLD|Same top|RMS' | ForEach-Object { $_.Line }

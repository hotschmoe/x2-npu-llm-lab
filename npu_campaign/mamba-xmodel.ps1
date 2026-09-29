# Same-session cross-model comparison with one package: Nemotron-3-Nano vs Qwen3.6-35B-A3B vs Gemma-4-26B-A4B (llama-bench, HTP0, interleaved rounds)
param([Parameter(Mandatory=$true)][string]$Pkg, [int]$Rounds = 2, [string]$Tag = 'xm')
. "$PSScriptRoot\mamba-std.ps1"
$repo = Split-Path $PSScriptRoot -Parent
$models = @(
  @('nemotron', "$repo\models\gguf\nemotron3-nano-30b-a3b-q4_0\nvidia_Nemotron-3-Nano-30B-A3B-Q4_0.gguf", 'GGML_HEXAGON_SSM_SCAN=2;GGML_HEXAGON_ROUTER_HVX=0'),
  @('qwen36',   "$repo\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", 'GGML_HEXAGON_ROUTER_HVX=1'),
  @('gemma4',   "$repo\models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf", 'GGML_HEXAGON_ROUTER_HVX=1'))
for ($r = 1; $r -le $Rounds; $r++) {
  foreach ($m in $models) {
    $log = "$Tag-$($m[0])-r$r"
    & "$PSScriptRoot\mamba-run.ps1" -Pkg $Pkg -Exe llama-bench -Log $log -TimeoutSec 2400 -Env "$MambaStdEnv;$($m[2])" -ArgLine "-m $($m[1]) -p 512,1024 -n 64 -b 2048 -ub 1024 -r 3 -ngl 99 --device HTP0 -o md" | Out-Null
    $rows = Get-Content "$PSScriptRoot\$log.log" | Where-Object { $_ -match '\|\s+(pp|tg)\d+' } | ForEach-Object { $c = $_ -split '\|'; ($c[8].Trim() + '=' + ($c[9] -replace '[^0-9. ]','' -replace '\s+',' ').Trim()) }
    "round $r $($m[0]): " + ($rows -join '  ')
  }
}

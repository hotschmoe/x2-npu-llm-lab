# bf16 agent: two traced runs of the same package (nondeterminism probe). Each takes the FIFO lock via trace-dump.ps1.
param([string]$Pkg = 'npu_campaign\pkg-clover-bf16-g', [string]$Tag = 'g', [string]$Filter = '^l_out-', [string]$EnvSpec = '')
$e = @{}
if ($EnvSpec) { foreach ($kv in $EnvSpec.Split(',')) { $k, $v = $kv.Split('='); $e[$k] = $v } }
foreach ($r in 1, 2) {
    & "$PSScriptRoot\trace-dump.ps1" -Label "bf16-$Tag-r$r" -Pkg $Pkg -PromptFile 'npu_campaign\clover-prompts\diffusion.txt' -DraftN 16 -Filter $Filter -Env $e
}

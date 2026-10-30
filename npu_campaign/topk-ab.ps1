# Interleaved A/B (base vs topk pkg), clover-run.ps1 holds the NPU lock per run (retries on lock-script errors).
# Usage: topk-ab.ps1 -Tag ab1 -Rounds 3 -PkgB npu_campaign\pkg-clover-topk-a [-EnvB @{CLOVER_TOPK_CUSTOM='1'}]
param([string]$Tag = 'ab1', [int]$Rounds = 3, [string]$PkgA = 'npu_campaign\pkg-clover-base', [string]$PkgB = 'npu_campaign\pkg-clover-topk-a',
      [hashtable]$EnvA = @{}, [hashtable]$EnvB = @{}, [string]$Exe = 'llama-speculative-simple.exe', [int]$NPredict = 256)
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($v in 'A', 'B') {
        $pkg = if ($v -eq 'A') { $PkgA } else { $PkgB }
        $e = if ($v -eq 'A') { $EnvA } else { $EnvB }
        "--- $Tag round $r variant $v ($pkg)"
        for ($try = 1; $try -le 30; $try++) {
            try {
                $ErrorActionPreference = 'Stop'
                & "$PSScriptRoot\clover-run.ps1" -Pkg $pkg -Label "$Tag-$v$r" -Device HTP0 -Accept '' -NPredict $NPredict -Exe $Exe -Env $e
                break
            } catch { "retry $try after: $($_.Exception.Message)"; . "$PSScriptRoot\topk-lockutil.ps1"; Remove-HalfLock; Start-Sleep -Seconds 2 }
        }
    }
}

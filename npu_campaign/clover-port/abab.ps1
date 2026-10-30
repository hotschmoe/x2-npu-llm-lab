# Interleaved A/B (ABAB...) of Clover draft-self runs. Each run goes through clover-run.ps1 (takes the NPU lock itself).
# Usage (dot-source or call with &): abab.ps1 -Tag x -Rounds 2 -Arms @(@{n='A';pkg='npu_campaign\pkg-clover-topk-c';env=@{}}, @{n='B';pkg='...';env=@{GGML_HEXAGON_GDN_SNAPDMA='0'}})
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][object[]]$Arms,
    [int]$Rounds = 2,
    [string]$Prompt = 'npu_campaign\clover-prompts\diffusion.txt',
    [int]$NPredict = 256,
    [string]$Extra = ''
)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$res = @()
for ($r = 1; $r -le $Rounds; $r++) {
    $order = if ($r % 2 -eq 1) { $Arms } else { @($Arms[($Arms.Count - 1)..0]) }
    foreach ($a in $order) {
        $label = "$Tag-$($a.n)-r$r"
        $out = & "$repo\npu_campaign\clover-run.ps1" -Label $label -Pkg $a.pkg -Accept '' -NPredict $NPredict -Prompt $Prompt -Env $a.env -Extra $Extra 2>&1 | Out-String
        $others = @(Get-Process -Name 'llama-*', 'test-backend-ops' -ErrorAction SilentlyContinue)
        $tps = $null; $cpu = $null
        $log = "$repo\npu_campaign\clover-logs\$label.err"
        $m = Select-String -Path $log -Pattern 'decoded +(\d+) tokens in +([\d.]+) seconds, speed: +([\d.]+)' | Select-Object -Last 1
        if ($m) { $tps = [double]$m.Matches[0].Groups[3].Value }
        $c = ($out -split "`n") | Where-Object { $_ -match 'HOST_CPU busy cores during generation = ([\d.]+)' } | Select-Object -First 1
        if ($c -match '= ([\d.]+)') { $cpu = [double]$Matches[1] }
        $row = [pscustomobject]@{ round = $r; arm = $a.n; tps = $tps; cores = $cpu; leftover_procs = $others.Count; label = $label }
        $res += $row
        "{0} r{1} {2}: {3} tok/s, host cores {4}, leftover llama procs {5}" -f $Tag, $r, $a.n, $tps, $cpu, $others.Count
    }
}
$res | Group-Object arm | ForEach-Object {
    $t = ($_.Group | Where-Object { $_.tps } | Measure-Object tps -Average).Average
    $c = ($_.Group | Where-Object { $_.cores } | Measure-Object cores -Average).Average
    "ARM {0}: mean tok/s {1:F3}  mean host cores {2:F2}  (n={3})" -f $_.Name, $t, $c, $_.Count
}

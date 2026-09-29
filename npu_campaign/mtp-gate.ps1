# Correctness gate for MTP speedup candidates: traced (full-vocab logits) runs of
# ref/cand x plain/mtp, then strict comparisons with compare_logits_traces.py.
# Deletes the .logits.bin files afterwards unless -Keep.
param(
    [Parameter(Mandatory = $true)][string]$Prefix,
    [string]$RefPkg = 'npu_campaign\pkg-mtp-base',
    [Parameter(Mandatory = $true)][string]$CandPkg,
    [string]$Prompt = 'Hello',
    [int]$Tokens = 128,
    [int]$DraftMax = 1,
    [string]$CandExtraEnv = '',
    [string]$Arms = 'ref-plain,cand-plain,ref-mtp,cand-mtp',
    [switch]$Keep
)
$ErrorActionPreference = 'Stop'
$run = Join-Path $PSScriptRoot 'run-mtp-speed.ps1'
$cmp = Join-Path $PSScriptRoot 'compare_logits_traces.py'
$dir = Join-Path $PSScriptRoot 'mtp-speed'
$armList = $Arms -split '[, ]+' | Where-Object { $_ }
foreach ($arm in $armList) {
    $pkg = if ($arm.StartsWith('ref')) { $RefPkg } else { $CandPkg }
    $spec = if ($arm.EndsWith('mtp')) { 'mtp' } else { 'none' }
    $extra = if ($arm.StartsWith('cand')) { $CandExtraEnv } else { '' }
    & $run -Label "$Prefix-$arm" -Pkg $pkg -Spec $spec -DraftMax $DraftMax -Prompt $Prompt -Tokens $Tokens -LogitsTrace -ExtraEnv $extra |
        Select-String 'RUN_EXIT|eval time|acceptance' | ForEach-Object { "  [$arm] $($_.Line.Trim())" }
}
function Cmp($a, $b, $extraArgs) {
    $pa = Join-Path $dir "$Prefix-$a.logits.bin"; $pb = Join-Path $dir "$Prefix-$b.logits.bin"
    if (-not ((Test-Path $pa) -and (Test-Path $pb))) { return }
    $out = & python $cmp $pa $pb --require-identical @extraArgs 2>&1
    "GATE $a vs $b $($extraArgs -join ' '): exit=$LASTEXITCODE :: " + (($out | Select-Object -Last 3) -join ' | ')
}
Cmp 'ref-plain' 'cand-plain' @()
Cmp 'ref-mtp' 'cand-mtp' @()
Cmp 'ref-mtp' 'cand-mtp' @('--context-index', '1', '--draft-greedy')
Cmp 'cand-plain' 'cand-mtp' @()
Cmp 'ref-plain' 'ref-mtp' @()
if (-not $Keep) { Get-ChildItem $dir -Filter "$Prefix-*.logits.bin" | Remove-Item }

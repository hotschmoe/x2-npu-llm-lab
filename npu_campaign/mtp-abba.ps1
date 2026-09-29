# Interleaved untraced speed comparison of MTP-speedup variants (ABBA / rotated order).
# -Variants: semicolon list of name=pkgdir|spec|draftmax|ExtraEnv  (ExtraEnv uses , between NAME=VAL pairs)
# e.g. "base=npu_campaign\pkg-mtp-base|mtp|1|;fc=npu_campaign\pkg-mtp-fc|mtp|1|LLAMA_MTP_FUSED_CATCHUP=1"
param(
    [Parameter(Mandatory = $true)][string]$Prefix,
    [Parameter(Mandatory = $true)][string]$Variants,
    [int]$Rounds = 2,
    [int]$Tokens = 256,
    [string]$Prompt = 'Hello'
)
$ErrorActionPreference = 'Stop'
$run = Join-Path $PSScriptRoot 'run-mtp-speed.ps1'
$dir = Join-Path $PSScriptRoot 'mtp-speed'
$vs = @()
foreach ($v in ($Variants -split ';' | Where-Object { $_ })) {
    $name, $rest = $v.Split('=', 2)
    $pkg, $spec, $dmax, $env = $rest.Split('|')
    $vs += [pscustomobject]@{ Name = $name; Pkg = $pkg; Spec = $spec; DraftMax = [int]$dmax; Env = ($env -replace ',', ';') }
}
$order = @()
for ($r = 0; $r -lt $Rounds; $r++) {
    $o = if ($r % 2 -eq 0) { $vs } else { [array]::Reverse(($c = @($vs))); $c }
    $order += $o
}
$results = @()
$i = 0
foreach ($v in $order) {
    $i++
    $label = "$Prefix-$($v.Name)-$i"
    & $run -Label $label -Pkg $v.Pkg -Spec $v.Spec -DraftMax $v.DraftMax -Tokens $Tokens -Prompt $Prompt -ExtraEnv $v.Env | Out-Null
    $log = Get-Content (Join-Path $dir "$label.detail.log") -Raw
    $m = [regex]::Match($log, '(?<!prompt )eval time =\s+([\d.]+) ms /\s+(\d+) tokens[^\n]*?([\d.]+) tokens per second')
    $a = [regex]::Match($log, 'draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated')
    $tps = [double]$m.Groups[3].Value
    $acc = if ($a.Success) { "$($a.Groups[2].Value)/$($a.Groups[3].Value)" } else { '-' }
    $results += [pscustomobject]@{ Run = $i; Variant = $v.Name; TokPerS = $tps; Accept = $acc }
    "RUN $i $($v.Name) tok/s=$tps accept=$acc"
    Remove-Item (Join-Path $dir "$label.log.err") -ErrorAction SilentlyContinue
}
"--- summary ($Prefix, $Tokens tokens, prompt '$Prompt')"
$results | Group-Object Variant | ForEach-Object {
    $t = $_.Group.TokPerS
    "{0,-12} mean {1,6:N2} tok/s  runs {2}" -f $_.Name, (($t | Measure-Object -Average).Average), (($t | ForEach-Object { '{0:N2}' -f $_ }) -join ' ')
}

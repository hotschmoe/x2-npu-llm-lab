# Same-build repeat protocol for the plain-path instability (decode2 campaign).
# Runs N identical traced plain runs of one package, hashes the full-vocab logit trace of each run, and reports
# distinct-hash counts for (a) the whole trace and (b) the prompt row only (row 0).
# Usage: repeat-plain.ps1 -Prefix rp1 -Pkg npu_campaign\pkg-decode2-base -N 8 [-Tokens 4] [-ExtraEnv "A=1;B=2"] [-Prompt Hello]
param(
    [Parameter(Mandatory = $true)][string]$Prefix,
    [Parameter(Mandatory = $true)][string]$Pkg,
    [int]$N = 8, [int]$Tokens = 4, [string]$Prompt = 'Hello', [string]$ExtraEnv = '', [string]$ExtraArgs = '', [switch]$Keep,
    [string]$ModelPath = 'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf'
)
$ErrorActionPreference = 'Stop'
$run = Join-Path $PSScriptRoot 'run-mtp-speed.ps1'
$dir = Join-Path $PSScriptRoot 'mtp-speed'
$sha = [System.Security.Cryptography.SHA256]::Create()
$rowBytes = $null
$res = @()
for ($i = 1; $i -le $N; $i++) {
    $lab = "$Prefix-$i"
    & $run -Label $lab -Pkg $Pkg -Spec none -Prompt $Prompt -Tokens $Tokens -LogitsTrace -ExtraEnv $ExtraEnv -ExtraArgs $ExtraArgs -ModelPath $ModelPath | Out-Null
    $f = Join-Path $dir "$lab.logits.bin"
    $bytes = [System.IO.File]::ReadAllBytes($f)
    $full = ([BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-', '').Substring(0, 12)
    # per-row hash: trace format is compare_logits_traces.py's; use the whole-file split into equal chunks of (len / (Tokens+1))? use python for exactness
    $res += [pscustomobject]@{ Run = $i; Full = $full; Size = $bytes.Length }
}
$res | Format-Table -AutoSize | Out-String | Write-Host
$d = ($res | Select-Object -ExpandProperty Full | Sort-Object -Unique).Count
$files = 1..$N | ForEach-Object { Join-Path $dir "$Prefix-$_.logits.bin" }
python (Join-Path $PSScriptRoot 'repeat_rows.py') @files
if (-not $Keep) { $files | Remove-Item }
"REPEAT $Prefix pkg=$Pkg env=[$ExtraEnv] N=$N distinct_full_hashes=$d"

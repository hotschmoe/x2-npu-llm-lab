param(
    [Parameter(Mandatory = $true)][string]$Prefix,
    [ValidateRange(1, 128)][int]$Tokens = 64,
    [string]$Prompt = 'Hello'
)

$ErrorActionPreference = 'Stop'
if ($Prefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid prefix' }
if ($Prompt -match '["\r\n]') { throw 'Prompt must be a single line without double quotes' }
if (Get-Process llama-cli,llama-bench,test-backend-ops,ninja -ErrorAction SilentlyContinue) { throw 'Build or inference already active' }

$repoRoot = Split-Path $PSScriptRoot -Parent
$installed = Join-Path $repoRoot 'llama.cpp-hexagon/pkg-wos/lib'
$hostDll = Join-Path $repoRoot 'llama.cpp-hexagon/pkg-wos/bin/ggml-hexagon.dll'
$candidate = Join-Path $PSScriptRoot 'backup-perop-wholecache-candidate'
$baseline = Join-Path $PSScriptRoot 'backup-perop-wholecache-prechange'
$model = Join-Path $repoRoot 'models/gguf/qwen36-35b-a3b-q4_0/Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf'
$files = @('libggml-htp-v81.so', 'libggml-htp.cat')
$baselineHash = @{
    'libggml-htp-v81.so' = 'D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD'
    'libggml-htp.cat' = '5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F'
}
$candidateHash = @{
    'libggml-htp-v81.so' = 'E9560F297EFB470BC70A61615BE6130F0E35CD6366FDE52A84F2C2D588FC5160'
    'libggml-htp.cat' = '54932DC01E4C62ECBBB8BB05315DEB8CD8B8FE4B9970E33EC147E038D66D713E'
}
if ((Get-FileHash -LiteralPath $hostDll -Algorithm SHA256).Hash -ne 'C9E4D8ADE6884DA62DF7A597DCEAC6582AC036236C06134231A405BAF8E6399E') { throw 'Unexpected host DLL' }
foreach ($name in $files) {
    if ((Get-FileHash -LiteralPath (Join-Path $installed $name) -Algorithm SHA256).Hash -ne $baselineHash[$name]) { throw "Unexpected installed baseline: $name" }
    if ((Get-FileHash -LiteralPath (Join-Path $baseline $name) -Algorithm SHA256).Hash -ne $baselineHash[$name]) { throw "Bad baseline archive: $name" }
    if ((Get-FileHash -LiteralPath (Join-Path $candidate $name) -Algorithm SHA256).Hash -ne $candidateHash[$name]) { throw "Bad candidate archive: $name" }
}
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash -ne 'E8E27D9475056E03B5D663C032B0966E00E68DB6314C07A883F791B9F6297746') { throw 'Unexpected model' }
foreach ($arm in @('plain-a', 'plain-b', 'real-a', 'real-b')) {
    foreach ($suffix in @('.log', '.log.err', '.detail.log', '.logits.bin')) {
        if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "$Prefix-$arm$suffix")) { throw "Existing artifacts: $Prefix-$arm" }
    }
}

try {
    foreach ($name in $files) {
        Copy-Item -LiteralPath (Join-Path $candidate $name) -Destination (Join-Path $installed $name) -Force
        if ((Get-FileHash -LiteralPath (Join-Path $installed $name) -Algorithm SHA256).Hash -ne $candidateHash[$name]) { throw "Candidate install mismatch: $name" }
    }
    foreach ($arm in @('plain-a', 'plain-b', 'real-a', 'real-b')) {
        $stageArgs = @{
            Label = "$Prefix-$arm"
            Tokens = $Tokens
            Prompt = $Prompt
            ModelPath = $model
            GdnSelect = 1
            FaSelect = 1
            RouterHvx = $true
            LogitsTrace = $true
        }
        if ($arm.StartsWith('real')) { $stageArgs.Mtp = $true }
        & (Join-Path $PSScriptRoot 'run-mtp-stage.ps1') @stageArgs
        if (-not $?) { throw "Stage failed: $arm" }
    }
} finally {
    if (Get-Process llama-cli,llama-bench,test-backend-ops -ErrorAction SilentlyContinue) { throw 'Inference remains live; preserving installed candidate for investigation' }
    foreach ($name in $files) {
        Copy-Item -LiteralPath (Join-Path $baseline $name) -Destination (Join-Path $installed $name) -Force
        if ((Get-FileHash -LiteralPath (Join-Path $installed $name) -Algorithm SHA256).Hash -ne $baselineHash[$name]) { throw "Baseline restoration mismatch: $name" }
    }
    Write-Output 'BASELINE_RESTORED=1'
}

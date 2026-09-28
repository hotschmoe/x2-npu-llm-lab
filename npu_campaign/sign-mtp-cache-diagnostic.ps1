param([ValidateSet('flush-only', 'activation-source')][string]$Variant = 'flush-only')

$ErrorActionPreference = 'Stop'
if (Get-Process llama-cli,llama-bench,test-backend-ops,ninja -ErrorAction SilentlyContinue) { throw 'Build or inference already active' }

$repoRoot = Split-Path $PSScriptRoot -Parent
$installed = Join-Path $repoRoot 'llama.cpp-hexagon/pkg-wos/lib'
$baseline = Join-Path $PSScriptRoot 'backup-perop-wholecache-prechange'
$build = Join-Path $PSScriptRoot "build-mtp-cache-$Variant"
$archive = Join-Path $PSScriptRoot "backup-mtp-cache-$Variant"
$sdkBin = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86'
$inf2cat = Join-Path $sdkBin 'inf2cat.exe'
$signtool = Join-Path $sdkBin 'signtool.exe'
$certificate = 'C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx'
$soName = 'libggml-htp-v81.so'
$catName = 'libggml-htp.cat'
$baselineHashes = @{
    $soName = 'D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD'
    $catName = '5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F'
}
$candidateHashes = @{
    'flush-only' = '3040B10D48740833359B00BD762C234420787C6D5CF7F0BA379B04AD091BF34F'
    'activation-source' = '64C095F085592FA6D30EE7E9A562549FEAF1ECD6997BD09AE065DC285C617993'
}
$candidateHash = $candidateHashes[$Variant]
foreach ($required in @($inf2cat, $signtool, $certificate)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing signer input: $required" }
}
foreach ($name in @($soName, $catName)) {
    foreach ($path in @((Join-Path $installed $name), (Join-Path $baseline $name))) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $baselineHashes[$name]) { throw "Unexpected baseline: $path" }
    }
}
if ((Get-FileHash -LiteralPath (Join-Path $build $soName) -Algorithm SHA256).Hash -ne $candidateHash) { throw 'Unexpected candidate build' }
if (Test-Path -LiteralPath $archive) { throw "Archive already exists: $archive" }

try {
    Copy-Item -LiteralPath (Join-Path $build $soName) -Destination (Join-Path $installed $soName) -Force
    if ((Get-FileHash -LiteralPath (Join-Path $installed $soName) -Algorithm SHA256).Hash -ne $candidateHash) { throw 'Candidate install mismatch' }
    & $inf2cat "/driver:$installed" '/os:10_25H2_ARM64'
    if ($LASTEXITCODE -ne 0) { throw 'Catalog creation failed' }
    & $signtool sign /fd sha256 /f $certificate (Join-Path $installed $catName)
    if ($LASTEXITCODE -ne 0) { throw 'Catalog signing failed' }
    & $signtool verify /pa /c (Join-Path $installed $catName) (Join-Path $installed $soName)
    if ($LASTEXITCODE -ne 0) { throw 'Catalog membership failed' }
    New-Item -ItemType Directory -Path $archive -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath (Join-Path $installed $soName) -Destination (Join-Path $archive $soName)
    Copy-Item -LiteralPath (Join-Path $installed $catName) -Destination (Join-Path $archive $catName)
    Write-Output "ARCHIVED=$Variant SO=$candidateHash CAT=$((Get-FileHash -LiteralPath (Join-Path $archive $catName) -Algorithm SHA256).Hash)"
} finally {
    if (Get-Process llama-cli,llama-bench,test-backend-ops -ErrorAction SilentlyContinue) { throw 'Inference remains live; preserving installed variant for investigation' }
    foreach ($name in @($soName, $catName)) {
        Copy-Item -LiteralPath (Join-Path $baseline $name) -Destination (Join-Path $installed $name) -Force
        if ((Get-FileHash -LiteralPath (Join-Path $installed $name) -Algorithm SHA256).Hash -ne $baselineHashes[$name]) { throw "Baseline restoration mismatch: $name" }
    }
    Write-Output 'BASELINE_RESTORED=1'
}

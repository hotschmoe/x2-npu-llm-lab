$ErrorActionPreference = 'Stop'
if (Get-Process llama-cli,llama-bench,test-backend-ops,ninja -ErrorAction SilentlyContinue) { throw 'Build or inference already active' }

$repoRoot = Split-Path $PSScriptRoot -Parent
$installed = Join-Path $repoRoot 'llama.cpp-hexagon/pkg-wos/lib'
$baseline = Join-Path $PSScriptRoot 'backup-perop-wholecache-prechange'
$sdkBin = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86'
$inf2cat = Join-Path $sdkBin 'inf2cat.exe'
$signtool = Join-Path $sdkBin 'signtool.exe'
$certificate = 'C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx'
$soName = 'libggml-htp-v81.so'
$catName = 'libggml-htp.cat'
$baselineSo = 'D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD'
$baselineCat = '5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F'
$variants = @(
    @{ Name = 'keep-tracker'; SoHash = '443B5B49D7576693245827D3BD315CF487B2303E4EBD460646A9E040CBF5591C' },
    @{ Name = 'reset-control'; SoHash = '8CE16D35BB1A8CE691BFBD6748DB809BB61E2BE102116511D22366CD35D6DD04' }
)

foreach ($required in @($inf2cat, $signtool, $certificate)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing signer input: $required" }
}
foreach ($name in @($soName, $catName)) {
    $expected = if ($name -eq $soName) { $baselineSo } else { $baselineCat }
    if ((Get-FileHash -LiteralPath (Join-Path $installed $name) -Algorithm SHA256).Hash -ne $expected) { throw "Unexpected installed baseline: $name" }
    if ((Get-FileHash -LiteralPath (Join-Path $baseline $name) -Algorithm SHA256).Hash -ne $expected) { throw "Bad baseline archive: $name" }
}
foreach ($variant in $variants) {
    $build = Join-Path $PSScriptRoot "build-mtp-cache-$($variant.Name)"
    $archive = Join-Path $PSScriptRoot "backup-mtp-cache-$($variant.Name)"
    if ((Get-FileHash -LiteralPath (Join-Path $build $soName) -Algorithm SHA256).Hash -ne $variant.SoHash) { throw "Bad build: $($variant.Name)" }
    if (Test-Path -LiteralPath $archive) { throw "Archive already exists: $archive" }
}

try {
    foreach ($variant in $variants) {
        $build = Join-Path $PSScriptRoot "build-mtp-cache-$($variant.Name)"
        $archive = Join-Path $PSScriptRoot "backup-mtp-cache-$($variant.Name)"
        Copy-Item -LiteralPath (Join-Path $build $soName) -Destination (Join-Path $installed $soName) -Force
        if ((Get-FileHash -LiteralPath (Join-Path $installed $soName) -Algorithm SHA256).Hash -ne $variant.SoHash) { throw "Install mismatch: $($variant.Name)" }

        & $inf2cat "/driver:$installed" '/os:10_25H2_ARM64'
        if ($LASTEXITCODE -ne 0) { throw "Catalog creation failed: $($variant.Name)" }
        & $signtool sign /fd sha256 /f $certificate (Join-Path $installed $catName)
        if ($LASTEXITCODE -ne 0) { throw "Catalog signing failed: $($variant.Name)" }
        & $signtool verify /pa /c (Join-Path $installed $catName) (Join-Path $installed $soName)
        if ($LASTEXITCODE -ne 0) { throw "Catalog membership failed: $($variant.Name)" }

        New-Item -ItemType Directory -Path $archive -ErrorAction Stop | Out-Null
        Copy-Item -LiteralPath (Join-Path $installed $soName) -Destination (Join-Path $archive $soName)
        Copy-Item -LiteralPath (Join-Path $installed $catName) -Destination (Join-Path $archive $catName)
        $signedCatHash = (Get-FileHash -LiteralPath (Join-Path $archive $catName) -Algorithm SHA256).Hash
        Write-Output "ARCHIVED=$($variant.Name) SO=$($variant.SoHash) CAT=$signedCatHash"
    }
} finally {
    if (Get-Process llama-cli,llama-bench,test-backend-ops -ErrorAction SilentlyContinue) { throw 'Inference remains live; preserving installed variant for investigation' }
    foreach ($name in @($soName, $catName)) {
        Copy-Item -LiteralPath (Join-Path $baseline $name) -Destination (Join-Path $installed $name) -Force
        $expected = if ($name -eq $soName) { $baselineSo } else { $baselineCat }
        if ((Get-FileHash -LiteralPath (Join-Path $installed $name) -Algorithm SHA256).Hash -ne $expected) { throw "Baseline restore mismatch: $name" }
    }
    Write-Output 'BASELINE_RESTORED=1'
}

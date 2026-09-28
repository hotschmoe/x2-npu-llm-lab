param(
    [Parameter(Mandatory = $true)][string]$Prefix,
    [ValidateRange(1, 128)][int]$Tokens = 64,
    [string]$Prompt = 'Hello',
    [switch]$NoLogitsTrace,
    [ValidateSet('archive', 'keep-tracker', 'reset-control', 'flush-only', 'activation-source', 'source-off', 'actsrc-64k', 'actsrc-4k', 'actsrc-small', 'actsrc-small-flush', 'actsrc-512', 'actsrc-513to4095', 'actsrc-small-inline', 'actsrc-small-inline-sync')][string]$Variant = 'archive'
)

$ErrorActionPreference = 'Stop'
if ($Prefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid prefix' }
if ($Prompt -match '["\r\n]') { throw 'Prompt must be a single line without double quotes' }
if (Get-Process llama-cli,llama-bench,test-backend-ops,ninja -ErrorAction SilentlyContinue) { throw 'Build or inference already active' }

$repoRoot = Split-Path $PSScriptRoot -Parent
$installed = Join-Path $repoRoot 'llama.cpp-hexagon/pkg-wos/lib'
$hostDll = Join-Path $repoRoot 'llama.cpp-hexagon/pkg-wos/bin/ggml-hexagon.dll'
$candidateName = if ($Variant -eq 'archive') { 'backup-perop-wholecache-candidate' } else { "backup-mtp-cache-$Variant" }
$candidate = Join-Path $PSScriptRoot $candidateName
$baseline = Join-Path $PSScriptRoot 'backup-perop-wholecache-prechange'
$model = Join-Path $repoRoot 'models/gguf/qwen36-35b-a3b-q4_0/Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf'
$files = @('libggml-htp-v81.so', 'libggml-htp.cat')
$baselineHash = @{
    'libggml-htp-v81.so' = 'D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD'
    'libggml-htp.cat' = '5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F'
}
$candidateHashes = @{
    'archive' = @{ 'libggml-htp-v81.so' = 'E9560F297EFB470BC70A61615BE6130F0E35CD6366FDE52A84F2C2D588FC5160'; 'libggml-htp.cat' = '54932DC01E4C62ECBBB8BB05315DEB8CD8B8FE4B9970E33EC147E038D66D713E' }
    'keep-tracker' = @{ 'libggml-htp-v81.so' = '443B5B49D7576693245827D3BD315CF487B2303E4EBD460646A9E040CBF5591C'; 'libggml-htp.cat' = '75C0C6585B85DA245BC15129FD3CF65C68AA0766E7B7DAA966F3E3EE1AA72E05' }
    'reset-control' = @{ 'libggml-htp-v81.so' = '8CE16D35BB1A8CE691BFBD6748DB809BB61E2BE102116511D22366CD35D6DD04'; 'libggml-htp.cat' = '7ECFCE6A864D4228BF11776777F86620A454BA726D985A7B4A9E49E7B366B093' }
    'flush-only' = @{ 'libggml-htp-v81.so' = '3040B10D48740833359B00BD762C234420787C6D5CF7F0BA379B04AD091BF34F'; 'libggml-htp.cat' = '3C916C976072B5F1F39BB0E27918673282C936B1371E922F75EA5E46828CA74B' }
    'activation-source' = @{ 'libggml-htp-v81.so' = '64C095F085592FA6D30EE7E9A562549FEAF1ECD6997BD09AE065DC285C617993'; 'libggml-htp.cat' = '4404E0B0BEEBEFFB956164FE0151017C9A01F6FC0F57ABEA6DCE279A5E3949B3' }
    'source-off' = @{ 'libggml-htp-v81.so' = 'CE61129577F7D63E3186ED1C439DF35F970954DE6BD23416FF1FD081C8A50CDA'; 'libggml-htp.cat' = 'BC4B72DF39B5E275AF2366D97172E38BB74D68CC535EDC1D9E0EF242898D5434' }
    'actsrc-64k' = @{ 'libggml-htp-v81.so' = '11CA04647743CE92068943B6B5D79D413C0CC2E3C95967FA76F2AA3386E9B21F'; 'libggml-htp.cat' = 'C6ACF1F139C81C3BF0B9241720004CD6EB66EDA528A1A0107B9142257544A831' }
    'actsrc-4k' = @{ 'libggml-htp-v81.so' = '9A05E3EA98E9B8D8C93C4D9B202F532BDAE8AA060D79E6A1FCB16802EE8E1064'; 'libggml-htp.cat' = '9CD2940201621536303A51CFE8BBA17E343437C9F1045BD7E708F53C422FD997' }
    'actsrc-small' = @{ 'libggml-htp-v81.so' = '143477CBE7650A9857AE37A2A6BDA2B3498209FC42B08C253ADF2E05D8CFABDC'; 'libggml-htp.cat' = 'F115149EC4F1716733AA7A7957959D45A19C2DC0D357C5A1B8839FCDCA9263DB' }
    'actsrc-small-flush' = @{ 'libggml-htp-v81.so' = '45B009B812E800B0ACA150F397720F3E61BE82E6A24036313198F385EE7A5458'; 'libggml-htp.cat' = '1DD75D33FCC67BED9B3B2A12E933F75E04311754756F62512308FE8722CA8648' }
    'actsrc-512' = @{ 'libggml-htp-v81.so' = '1C9FB5DAFD40B8DF5F1C935C1D2B9937694FA448DAAED638B3B7CF79D35E5D69'; 'libggml-htp.cat' = '91A301E07F80935F778D78DA2ADA24FF763860A8B98E96C40A05CFAA71887798' }
    'actsrc-513to4095' = @{ 'libggml-htp-v81.so' = '4CDFE90088815E959B5354EFAC8D3FF9BEA1B913FC6ED905E94E5EC4C5A2E60A'; 'libggml-htp.cat' = 'D0608A3417A2710A06F57C0B020ED1686B379AC78B845FD8BCF960A6139B95FC' }
    'actsrc-small-inline' = @{ 'libggml-htp-v81.so' = '13A92F2145E5A56CA277B3B2CACBFEFAB42D87BF77932A70CFB56140C1A1AAE6'; 'libggml-htp.cat' = '2F835AA594EF5690DF06F76647AA0EE30C3AE2B02EF705DB17101B0040FD7287' }
    'actsrc-small-inline-sync' = @{ 'libggml-htp-v81.so' = '77BAE4219AD65135163C2D72DBA97FB0D539C90479993F97EC480244966DF9DC'; 'libggml-htp.cat' = 'E6750D7F17FF02F4B2C79B28CAC1C261702F23A55C9F10D1DBF524F36DD2B5B9' }
}
$candidateHash = $candidateHashes[$Variant]
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
        }
        if (-not $NoLogitsTrace) { $stageArgs.LogitsTrace = $true }
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

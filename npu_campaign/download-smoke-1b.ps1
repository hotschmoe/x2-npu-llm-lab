$ErrorActionPreference = 'Stop'
$dest = 'C:\Users\hotschmoe\Documents\llm_inference\models\gguf\llama32-1b-instruct-q4_0'
$log = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\smoke-download.log'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
function Say($msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
    Write-Host $line
}
$repo = 'bartowski/Llama-3.2-1B-Instruct-GGUF'
$file = 'Llama-3.2-1B-Instruct-Q4_0.gguf'
Say "START $file"
& hf download $repo $file --local-dir $dest
if ($LASTEXITCODE) {
    Say "FAILED exit $LASTEXITCODE"
    exit $LASTEXITCODE
}
$path = Join-Path $dest $file
$len = (Get-Item -LiteralPath $path).Length
Say "DONE bytes=$len"
exit 0

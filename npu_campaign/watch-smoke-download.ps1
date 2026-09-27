$ErrorActionPreference = 'Continue'
$log = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\smoke-download.log'
for ($i = 0; $i -lt 40 -and -not (Test-Path -LiteralPath $log); $i++) {
    Start-Sleep -Seconds 15
}
if (-not (Test-Path -LiteralPath $log)) {
    Write-Output 'FAILED: smoke-download.log never appeared'
    exit 1
}
while ($true) {
    $text = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue
    if ($text -match 'FAILED') {
        Write-Output 'FAILED: smoke model download'
        exit 1
    }
    if ($text -match 'DONE bytes=') {
        Write-Output 'DONE: smoke model download'
        exit 0
    }
    Start-Sleep -Seconds 15
}

$ErrorActionPreference = 'Continue'
$log = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\build.log'
$trace = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\watch-build.trace'
function Note($msg) {
    Add-Content -Path $trace -Value ('{0} {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg)
}
Note 'watch start'
for ($i = 0; $i -lt 40 -and -not (Test-Path -LiteralPath $log); $i++) {
    Start-Sleep -Seconds 30
}
if (-not (Test-Path -LiteralPath $log)) {
    Write-Output 'FAILED: build.log never appeared'
    exit 1
}
while ($true) {
    $hit = Select-String -Path $log -Pattern '^BUILD_EXIT=(\d+)' | Select-Object -Last 1
    if ($hit) {
        $code = $hit.Matches[0].Groups[1].Value
        Note "saw BUILD_EXIT=$code"
        if ($code -eq '0') {
            Write-Output 'DONE: hexagon build exit 0'
            exit 0
        }
        Write-Output "FAILED: hexagon build BUILD_EXIT=$code"
        exit 1
    }
    Start-Sleep -Seconds 30
}

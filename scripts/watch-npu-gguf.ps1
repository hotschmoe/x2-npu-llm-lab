$log = 'C:\Users\hotschmoe\Documents\llm_inference\models\gguf\download.log'
while ($true) {
    $text = ''
    if (Test-Path $log) {
        $text = Get-Content $log -Raw -ErrorAction SilentlyContinue
        if ($text -match 'ALL_DONE') { Write-Output 'DONE'; exit 0 }
        if ($text -match 'FAILED') { Write-Output 'FAILED'; exit 1 }
    }
    $alive = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'download-npu-gguf' }
    if (-not $alive -and $text -and $text -notmatch 'ALL_DONE') {
        Write-Output 'FAILED process exited'
        exit 1
    }
    Start-Sleep -Seconds 30
}

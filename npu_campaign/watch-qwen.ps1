$ErrorActionPreference = 'Continue'
$log = 'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\htp-qwen-profile1.log'
for ($i = 0; $i -lt 40 -and -not (Test-Path -LiteralPath $log); $i++) { Start-Sleep -Seconds 15 }
if (-not (Test-Path -LiteralPath $log)) {
    Write-Output 'FAILED: qwen profile log never appeared'
    exit 1
}
while ($true) {
    $hit = Select-String -Path $log -Pattern 'eval time|QWEN_EXIT|error:|failed to' -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($hit -and $hit.Line -match 'eval time') {
        Write-Output 'DONE: qwen profile-1 has eval time'
        exit 0
    }
    $proc = Get-Process llama-completion -ErrorAction SilentlyContinue
    if (-not $proc -and $hit -and $hit.Line -match 'failed|error:') {
        Write-Output ("FAILED: " + $hit.Line)
        exit 1
    }
    if (-not $proc -and ((Get-Item $log).Length -gt 200)) {
        $tail = Get-Content -Tail 5 -LiteralPath $log -ErrorAction SilentlyContinue
        $joined = ($tail -join ' ')
        if ($joined -match 'eval time') {
            Write-Output 'DONE: qwen profile-1 has eval time'
            exit 0
        }
        Write-Output 'FAILED: llama-completion exited before eval time'
        exit 1
    }
    Start-Sleep -Seconds 30
}

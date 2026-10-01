# Smoke test: start chat-mode server, run 3 PI headless tool-call cases (npu_campaign\pitest\run_suite.py, external server), print pass/fail and speeds, stop.
# Holds the NPU lock for the duration (~5-10 min). Usage: serve\smoke.ps1 [-Cases create_file,read_answer,shell_cmd] [-Port 8080]
param([string]$Cases = 'create_file,read_answer,shell_cmd', [int]$Port = 8080, [string]$Mode = 'chat')
$root = Split-Path $PSScriptRoot -Parent
& "$PSScriptRoot\start-qwen.ps1" -Mode $Mode -Port $Port
if ($LASTEXITCODE -ne 0) { "server did not start"; exit 1 }
$rc = 1
try {
    $tag = "smoke-$(Get-Date -Format yyyyMMdd-HHmmss)"
    & python "$root\npu_campaign\pitest\run_suite.py" --model qwen --reps 1 --cases $Cases --server-url "http://127.0.0.1:$Port" --tag $tag --ctx 32768 | Tee-Object -Variable out
    $fails = ($out | Select-String ': FAIL').Count
    $m = (Invoke-WebRequest "http://127.0.0.1:$Port/metrics" -UseBasicParsing).Content
    foreach ($k in 'prompt_tokens_seconds', 'predicted_tokens_seconds', 'tokens_predicted_total', 'prompt_tokens_total') {
        $l = ($m -split "`n" | Where-Object { $_ -match "^llamacpp:$k " }); if ($l) { "$k = $(($l -split ' ')[1])" }
    }
    if ($Mode -ne 'chat') { $l = ($m -split "`n" | Where-Object { $_ -match '^llamacpp:spec_decode_num_(draft|accepted)_tokens_total ' }); $l }
    "SMOKE: $(if ($fails -eq 0) { 'PASS' } else { "FAIL ($fails cases)" })"
    $rc = [int]($fails -ne 0)
} finally { & "$PSScriptRoot\stop-qwen.ps1" }
exit $rc

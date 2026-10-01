# Health check: prints server status, mode, lock owner, and a 1-token completion round trip.
param([int]$Port = 8080)
$root = Split-Path $PSScriptRoot -Parent
try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3; "health: $($h.status)" } catch { "health: DOWN ($($_.Exception.Message))"; exit 1 }
"mode: $(Get-Content (Join-Path $PSScriptRoot 'run\mode.txt') -ErrorAction SilentlyContinue)"
"lock: $(Get-Content (Join-Path $root 'npu_campaign\.npu.lock\owner.txt') -ErrorAction SilentlyContinue)"
$r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'
"reply: $($r.choices[0].message.content)"

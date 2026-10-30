# Qwen3.6 plain decode on HTP0 (no CPU fallback ops): is the host thread idle while waiting for the NPU?
$d = $PSScriptRoot
& "$d\thr.ps1" -Label qwen-t8 -Qwen -NPredict 300 -Threads '-t 8 -tb 8'

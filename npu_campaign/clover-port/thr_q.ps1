# host-thread CPU census: default threads vs few threads / no polling (same pkg), one lock hold each
$d = $PSScriptRoot
& "$d\thr.ps1" -Label t8 -Threads '-t 8 -tb 8' -NPredict 128
& "$d\thr.ps1" -Label t2poll0 -Threads '-t 2 -tb 2 --poll 0' -NPredict 128
& "$d\thr.ps1" -Label t1 -Threads '-t 1 -tb 1' -NPredict 128

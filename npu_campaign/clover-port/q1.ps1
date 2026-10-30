# queue 1: op-level gates and perf for the port packages (each step takes the NPU lock separately)
$ErrorActionPreference = 'Continue'
$d = $PSScriptRoot
$all = 'npu_campaign\pkg-clover-port-all'
& "$d\tbo.ps1" -Label all-gdn-test -Pkg $all -Mode test -Op GATED_DELTA_NET
& "$d\tbo.ps1" -Label all-gdn-test-snap0 -Pkg $all -Mode test -Op GATED_DELTA_NET -Env @{GGML_HEXAGON_GDN_SNAPDMA='0'}
foreach ($r in 1, 2) {
    & "$d\tbo.ps1" -Label "all-gdn-perf-snap1-r$r" -Pkg $all -Mode perf -Op GATED_DELTA_NET -Filter 'head_count=32,head_size=128,n_seq_tokens=16' -Env @{GGML_HEXAGON_GDN_SNAPDMA='1'}
    & "$d\tbo.ps1" -Label "all-gdn-perf-snap0-r$r" -Pkg $all -Mode perf -Op GATED_DELTA_NET -Filter 'head_count=32,head_size=128,n_seq_tokens=16' -Env @{GGML_HEXAGON_GDN_SNAPDMA='0'}
}
foreach ($r in 1, 2) {
    foreach ($p in 'dr', 'pad') {
        & "$d\tbo.ps1" -Label "$p-mmid-perf-r$r" -Pkg "npu_campaign\pkg-clover-port-$p" -Mode perf -Op MUL_MAT_ID -Filter 'n_mats=256,n_used=8'
        & "$d\tbo.ps1" -Label "$p-mm-perf-r$r" -Pkg "npu_campaign\pkg-clover-port-$p" -Mode perf -Op MUL_MAT -Filter 'm=8192,n=(1|4|16),k=2048'
    }
}

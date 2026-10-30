# queue 2: GDN cache-fusion snapshot test (exercises the SNAP kernel with dst_cache) + baseline GDN failure list on topk-c
$d = $PSScriptRoot
& "$d\tbo.ps1" -Label all-gdncf-test -Pkg 'npu_campaign\pkg-clover-port-all' -Mode test -Op GATED_DELTA_NET_CACHE_FUSION
& "$d\tbo.ps1" -Label all-gdncf-test-snap0 -Pkg 'npu_campaign\pkg-clover-port-all' -Mode test -Op GATED_DELTA_NET_CACHE_FUSION -Env @{GGML_HEXAGON_GDN_SNAPDMA='0'}
& "$d\tbo.ps1" -Label topkc-gdn-test -Pkg 'npu_campaign\pkg-clover-topk-c' -Mode test -Op GATED_DELTA_NET

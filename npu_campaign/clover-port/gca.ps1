$pre = 'npu_campaign\pkg-clover-port-'
$G1 = @{ n = 'gc1'; pkg = "${pre}gc"; env = @{ GGML_HEXAGON_HOSTSTAT = '1' } }
$G2 = @{ n = 'gc4096'; pkg = "${pre}gc"; env = @{ GGML_HEXAGON_HOSTSTAT = '1'; GGML_HEXAGON_GRAPH_CACHE = '4096' } }
& "$PSScriptRoot\blk.ps1" -Tag gca -Seq @($G1, $G2, $G2, $G1)

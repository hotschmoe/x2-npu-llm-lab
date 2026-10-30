$B  = @{ n = 'all'; pkg = 'npu_campaign\pkg-clover-port-all'; env = @{} }
$O1 = @{ n = 'allpoll'; pkg = 'npu_campaign\pkg-clover-port-all'; env = @{ GGML_HEXAGON_OPPOLL = '1' } }
& "$PSScriptRoot\bench.ps1" -Tag b2 -Seq @($B, $O1, $O1, $B)
